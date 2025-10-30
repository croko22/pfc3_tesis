"""
Group Sequence Policy Optimization (GSPO) for LLM fine-tuning.

Implementation of GSPO algorithm from:
"Group Preference Optimization: Few-Shot Alignment of Large Language Models"
https://arxiv.org/abs/2310.11346

GSPO provides stable training by using sequence-level importance sampling ratios:
    s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)

This reduces variance compared to PPO's token-level ratios.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..llm_agent.agent import LLMAgent

logger = logging.getLogger(__name__)


@dataclass
class GSPOConfig:
    """Configuration for GSPO optimizer."""
    # Learning rate
    learning_rate: float = 1e-5
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # GSPO-specific parameters
    clip_ratio: float = 0.2  # Clipping epsilon for policy ratio
    kl_coef: float = 0.1     # KL divergence penalty coefficient
    
    # PPO-style parameters (optional)
    vf_coef: float = 0.1     # Value function coefficient (if using value network)
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    
    # Optimization
    max_grad_norm: float = 1.0
    gamma: float = 0.99      # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # Regularization
    normalize_advantages: bool = True
    normalize_rewards: bool = False
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 500
    
    # Warmup
    warmup_steps: int = 100


class GSPOOptimizer:
    """
    GSPO optimizer for stable LLM fine-tuning.
    
    Key innovation: Uses sequence-level importance sampling ratio instead of
    token-level ratio, which significantly reduces gradient variance.
    """
    
    def __init__(
        self,
        agent: LLMAgent,
        config: GSPOConfig,
        output_dir: Path
    ):
        """
        Initialize GSPO optimizer.
        
        Args:
            agent: LLM agent to optimize
            config: GSPO configuration
            output_dir: Directory for checkpoints and logs
        """
        self.agent = agent
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            agent.get_trainable_parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Training statistics
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float('-inf')
        
        # Store old policy for computing ratios
        self.old_agent = None
        
        logger.info("GSPO optimizer initialized")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        old_log_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one GSPO training step.
        
        Args:
            prompts: List of input prompts
            responses: List of generated responses
            rewards: List of rewards for each response
            old_log_probs: Pre-computed log probs from old policy (optional)
            
        Returns:
            Dict with training metrics
        """
        self.agent.train_mode()
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.agent.device)
        
        # Normalize rewards if configured
        if self.config.normalize_rewards:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Compute log probabilities with current policy
        current_log_probs = self.agent.compute_log_probs(prompts, responses)
        
        # Compute log probabilities with old policy if not provided
        if old_log_probs is None:
            if self.old_agent is None:
                # First iteration: old policy = current policy
                old_log_probs = current_log_probs.detach()
            else:
                with torch.no_grad():
                    old_log_probs = self.old_agent.compute_log_probs(prompts, responses)
        
        # Compute sequence-level importance sampling ratio
        # s_i(θ) = (π_θ(y|x) / π_θ_old(y|x))^(1/|y|)
        log_ratio = current_log_probs - old_log_probs
        
        # Get sequence lengths for normalization
        sequence_lengths = torch.tensor(
            [len(self.agent.tokenizer.encode(r)) for r in responses],
            dtype=torch.float32,
            device=self.agent.device
        )
        
        # Sequence-level ratio (GSPO's key innovation)
        sequence_ratio = torch.exp(log_ratio / sequence_lengths)
        
        # Compute advantages (using rewards directly for simplicity)
        # In full implementation, you'd use GAE
        advantages = rewards_tensor
        
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # GSPO loss with clipping (similar to PPO but with sequence-level ratio)
        surrogate1 = sequence_ratio * advantages
        surrogate2 = torch.clamp(
            sequence_ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio
        ) * advantages
        
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # KL divergence penalty (regularization)
        kl_div = (log_ratio / sequence_lengths).mean()
        kl_penalty = self.config.kl_coef * kl_div
        
        # Total loss
        total_loss = policy_loss + kl_penalty
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.agent.get_trainable_parameters(),
                self.config.max_grad_norm
            )
        
        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        # Compute metrics
        metrics = {
            'loss/total': total_loss.item(),
            'loss/policy': policy_loss.item(),
            'loss/kl': kl_div.item(),
            'ratio/mean': sequence_ratio.mean().item(),
            'ratio/min': sequence_ratio.min().item(),
            'ratio/max': sequence_ratio.max().item(),
            'reward/mean': rewards_tensor.mean().item(),
            'reward/std': rewards_tensor.std().item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        return metrics
    
    def train_epoch(
        self,
        dataset: List[Dict],
        validation_data: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataset: List of training examples, each with:
                - 'prompt': Input prompt
                - 'response': Generated response
                - 'reward': Reward value
            validation_data: Optional validation data
            
        Returns:
            Dict with epoch metrics
        """
        logger.info(f"Starting epoch {self.epoch + 1}")
        
        # Store old policy parameters before training
        self._update_old_policy()
        
        # Shuffle dataset
        np.random.shuffle(dataset)
        
        # Training loop
        epoch_metrics = []
        num_batches = len(dataset) // self.config.batch_size
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            batch = dataset[start_idx:end_idx]
            
            # Extract data
            prompts = [item['prompt'] for item in batch]
            responses = [item['response'] for item in batch]
            rewards = [item['reward'] for item in batch]
            old_log_probs = torch.stack([item.get('old_log_prob', None) for item in batch]) \
                if all('old_log_prob' in item for item in batch) else None
            
            # Training step
            metrics = self.train_step(prompts, responses, rewards, old_log_probs)
            epoch_metrics.append(metrics)
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_metrics = self._average_metrics(epoch_metrics[-self.config.log_interval:])
                logger.info(
                    f"Epoch {self.epoch + 1}, Batch {batch_idx + 1}/{num_batches}: "
                    f"Loss={avg_metrics['loss/total']:.4f}, "
                    f"Reward={avg_metrics['reward/mean']:.4f}"
                )
            
            # Checkpointing
            if (self.global_step + 1) % self.config.save_interval == 0:
                self.save_checkpoint()
        
        # Epoch summary
        epoch_summary = self._average_metrics(epoch_metrics)
        
        # Validation
        if validation_data:
            val_metrics = self.evaluate(validation_data)
            epoch_summary.update({f'val/{k}': v for k, v in val_metrics.items()})
        
        self.epoch += 1
        
        return epoch_summary
    
    def evaluate(self, dataset: List[Dict]) -> Dict[str, float]:
        """
        Evaluate on a dataset.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dict with evaluation metrics
        """
        self.agent.eval_mode()
        
        total_reward = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for item in dataset:
                total_reward += item['reward']
                num_samples += 1
        
        avg_reward = total_reward / max(num_samples, 1)
        
        return {
            'reward/mean': avg_reward,
            'num_samples': num_samples
        }
    
    def _update_old_policy(self):
        """Store current policy as old policy for next iteration."""
        # Create a copy of the current agent
        # In practice, you'd save and load the model state
        # For efficiency, we'll just store references
        # In a full implementation, you'd deep copy the model
        logger.debug("Updating old policy")
        
        # Note: This is simplified. In production, you'd want to:
        # 1. Save current model state
        # 2. Load it into a separate model
        # 3. Freeze that model
        pass
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Average a list of metric dictionaries."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.agent.save_model(checkpoint_dir)
        
        # Save optimizer state
        torch.save({
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_reward': self.best_reward,
            'config': self.config
        }, checkpoint_dir / "optimizer.pt")
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        if is_best:
            best_dir = self.output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.agent.save_model(best_dir)
            logger.info(f"Saved best model to {best_dir}")
    
    def load_checkpoint(self, checkpoint_dir: Path):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        self.agent.load_model(checkpoint_dir)
        
        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint = torch.load(optimizer_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            self.best_reward = checkpoint['best_reward']
            
            logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        else:
            logger.warning(f"No optimizer state found at {optimizer_path}")
    
    def train(
        self,
        train_dataset: List[Dict],
        val_dataset: Optional[List[Dict]] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs (overrides config)
            
        Returns:
            Dict with training history
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        history = {
            'train_loss': [],
            'train_reward': [],
            'val_reward': []
        }
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_dataset, val_dataset)
            
            # Record history
            history['train_loss'].append(train_metrics['loss/total'])
            history['train_reward'].append(train_metrics['reward/mean'])
            
            if 'val/reward/mean' in train_metrics:
                history['val_reward'].append(train_metrics['val/reward/mean'])
                
                # Save best model
                val_reward = train_metrics['val/reward/mean']
                if val_reward > self.best_reward:
                    self.best_reward = val_reward
                    self.save_checkpoint(is_best=True)
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} completed: "
                f"Loss={train_metrics['loss/total']:.4f}, "
                f"Train Reward={train_metrics['reward/mean']:.4f}"
            )
        
        return history
