"""
LLM Agent with LoRA for test refinement.

Implements the policy (LLM) that generates refined tests.
"""

import logging
import torch
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import re

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM-based agent for test refinement with LoRA.
    
    This agent acts as the policy in the RL framework, generating refined
    test code given a prompt.
    """
    
    def __init__(
        self,
        model_name: str,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        generation_config: Optional[Dict] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize LLM agent.
        
        Args:
            model_name: HuggingFace model name or path
            use_lora: Whether to use LoRA
            lora_config: LoRA configuration dict
            generation_config: Generation parameters
            device: Device to use ('cuda' or 'cpu')
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora
        
        # Default configs
        self.lora_config = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        self.generation_config = generation_config or {
            "max_length": 2048,
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": 1
        }
        
        logger.info(f"Initializing LLM agent with model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Apply LoRA if requested
        if use_lora:
            logger.info("Applying LoRA to model")
            
            # Prepare model for k-bit training if using quantization
            if load_in_4bit or load_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Create LoRA config
            peft_config = LoraConfig(**self.lora_config)
            
            # Apply LoRA
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Set model to training mode for optimization
        self.model.train()
        
        logger.info("LLM agent initialized successfully")
    
    def generate(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[str]:
        """
        Generate refined test code from prompts.
        
        Args:
            prompts: List of prompts
            **generation_kwargs: Override generation config
            
        Returns:
            List of generated test codes
        """
        # Merge generation configs
        gen_config = {**self.generation_config, **generation_kwargs}
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_config.get("max_length", 2048)
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 1024),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9),
                do_sample=gen_config.get("do_sample", True),
                num_return_sequences=gen_config.get("num_return_sequences", 1),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Extract only the new generation (remove prompt)
        refined_codes = []
        for prompt, generated in zip(prompts, generated_texts):
            # Remove the prompt part
            if generated.startswith(prompt):
                refined = generated[len(prompt):].strip()
            else:
                refined = generated
            
            # Extract code from markdown if present
            refined = self._extract_code_from_response(refined)
            refined_codes.append(refined)
        
        return refined_codes
    
    def generate_with_logprobs(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate with log probabilities (for RL training).
        
        Args:
            prompts: List of prompts
            **generation_kwargs: Override generation config
            
        Returns:
            Tuple of (generated_codes, log_probs)
        """
        # Merge generation configs
        gen_config = {**self.generation_config, **generation_kwargs}
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_config.get("max_length", 2048)
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        
        # Generate with output_scores
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 1024),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            do_sample=gen_config.get("do_sample", True),
            num_return_sequences=gen_config.get("num_return_sequences", 1),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Get generated sequences
        sequences = outputs.sequences
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True
        )
        
        # Extract code
        refined_codes = []
        for prompt, generated in zip(prompts, generated_texts):
            if generated.startswith(prompt):
                refined = generated[len(prompt):].strip()
            else:
                refined = generated
            refined = self._extract_code_from_response(refined)
            refined_codes.append(refined)
        
        # Calculate log probabilities
        # Stack scores (list of tensors) -> (seq_len, batch_size, vocab_size)
        scores = torch.stack(outputs.scores, dim=0)
        
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
        
        # Get log probs of generated tokens
        generated_tokens = sequences[:, input_length:]  # Remove prompt tokens
        
        # Gather log probs of actual generated tokens
        token_log_probs = torch.gather(
            log_probs.permute(1, 0, 2),  # (batch, seq_len, vocab)
            dim=2,
            index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probs for each sequence
        sequence_log_probs = token_log_probs.sum(dim=1)
        
        return refined_codes, sequence_log_probs
    
    def compute_log_probs(
        self,
        prompts: List[str],
        completions: List[str]
    ) -> torch.Tensor:
        """
        Compute log probabilities of completions given prompts.
        
        Used for computing policy ratios in GSPO.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            
        Returns:
            Log probabilities tensor
        """
        # Combine prompts and completions
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        # Tokenize
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.generation_config.get("max_length", 2048)
        )
        
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.generation_config.get("max_length", 2048)
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get completion tokens
        labels = inputs["input_ids"][:, prompt_length:]
        
        # Gather log probs of actual tokens
        token_log_probs = torch.gather(
            log_probs[:, prompt_length - 1:-1, :],  # Shift by 1 for next-token prediction
            dim=2,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probs for each sequence (accounting for padding)
        attention_mask = inputs["attention_mask"][:, prompt_length:]
        sequence_log_probs = (token_log_probs * attention_mask).sum(dim=1)
        
        return sequence_log_probs
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Java code from LLM response.
        
        Handles markdown code blocks and other formatting.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted Java code
        """
        # Try to extract from markdown code block
        code_block_pattern = r'```(?:java)?\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code block, look for class definition
        class_pattern = r'(package\s+.*?|import\s+.*?|public\s+class\s+.*?\{.*)'
        matches = re.search(class_pattern, response, re.DOTALL)
        
        if matches:
            return matches.group(0).strip()
        
        # Return as-is if no patterns match
        return response.strip()
    
    def save_model(self, path: Path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.use_lora:
            # Save only LoRA adapters
            self.model.save_pretrained(path)
            logger.info(f"Saved LoRA adapters to {path}")
        else:
            # Save full model
            self.model.save_pretrained(path)
            logger.info(f"Saved model to {path}")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        
        if self.use_lora:
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                self.model,
                path,
                is_trainable=True
            )
            logger.info(f"Loaded LoRA adapters from {path}")
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info(f"Loaded model from {path}")
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
