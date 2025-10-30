"""
Reward function for test refinement.

Multi-objective reward that balances maintainability and effectiveness.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

from ..evaluation.metrics import TestMetrics

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward function weights and normalization."""
    # Weights for different components
    w_test_smells: float = -0.3
    w_cyclomatic_complexity: float = -0.2
    w_readability: float = 0.15
    w_branch_coverage: float = 0.25
    w_mutation_score: float = 0.25
    w_compilation: float = 0.15
    
    # Penalties
    failure_penalty: float = -1.0
    
    # Normalization bounds
    complexity_min: float = 1.0
    complexity_max: float = 50.0
    smells_min: float = 0.0
    smells_max: float = 10.0


class RewardFunction:
    """
    Multi-objective reward function for test refinement.
    
    Combines maintainability and effectiveness metrics to guide LLM optimization.
    """
    
    def __init__(self, config: RewardConfig):
        """
        Initialize reward function.
        
        Args:
            config: Reward configuration
        """
        self.config = config
    
    def calculate_reward(
        self,
        original_metrics: TestMetrics,
        refined_metrics: TestMetrics,
        normalization: bool = True
    ) -> float:
        """
        Calculate reward for a refined test.
        
        The reward is designed to:
        1. Preserve or improve effectiveness (coverage, mutation score)
        2. Improve maintainability (reduce smells, complexity)
        3. Ensure compilation and execution success
        
        Args:
            original_metrics: Metrics from EvoSuite-generated test
            refined_metrics: Metrics from LLM-refined test
            normalization: Whether to normalize metrics
            
        Returns:
            Reward value (higher is better)
        """
        # 1. Check critical failures
        if not refined_metrics.compiles:
            logger.debug("Test doesn't compile - applying failure penalty")
            return self.config.failure_penalty
        
        if not refined_metrics.passes:
            logger.debug("Test doesn't pass - applying failure penalty")
            return self.config.failure_penalty * 0.5  # Less severe than compilation failure
        
        # 2. Calculate individual reward components
        reward_components = {}
        
        # Maintainability rewards
        reward_components['smells'] = self._calculate_smell_reward(
            original_metrics, refined_metrics, normalization
        )
        
        reward_components['complexity'] = self._calculate_complexity_reward(
            original_metrics, refined_metrics, normalization
        )
        
        reward_components['readability'] = self._calculate_readability_reward(
            original_metrics, refined_metrics
        )
        
        # Effectiveness rewards
        reward_components['coverage'] = self._calculate_coverage_reward(
            original_metrics, refined_metrics
        )
        
        reward_components['mutation'] = self._calculate_mutation_reward(
            original_metrics, refined_metrics
        )
        
        # Compilation bonus
        reward_components['compilation'] = self.config.w_compilation
        
        # 3. Combine weighted rewards
        total_reward = (
            self.config.w_test_smells * reward_components['smells'] +
            self.config.w_cyclomatic_complexity * reward_components['complexity'] +
            self.config.w_readability * reward_components['readability'] +
            self.config.w_branch_coverage * reward_components['coverage'] +
            self.config.w_mutation_score * reward_components['mutation'] +
            reward_components['compilation']
        )
        
        logger.debug(f"Reward components: {reward_components}")
        logger.debug(f"Total reward: {total_reward}")
        
        return total_reward
    
    def _calculate_smell_reward(
        self,
        original: TestMetrics,
        refined: TestMetrics,
        normalize: bool
    ) -> float:
        """
        Calculate reward based on test smell reduction.
        
        Positive reward for reducing smells.
        
        Args:
            original: Original metrics
            refined: Refined metrics
            normalize: Whether to normalize
            
        Returns:
            Smell reward (positive = fewer smells)
        """
        original_smells = sum(original.test_smells.values())
        refined_smells = sum(refined.test_smells.values())
        
        # Calculate improvement
        smell_reduction = original_smells - refined_smells
        
        if normalize:
            # Normalize to [0, 1] range
            smell_reduction = self._normalize(
                smell_reduction,
                -self.config.smells_max,  # Worst case: all smells added
                self.config.smells_max    # Best case: all smells removed
            )
        
        return smell_reduction
    
    def _calculate_complexity_reward(
        self,
        original: TestMetrics,
        refined: TestMetrics,
        normalize: bool
    ) -> float:
        """
        Calculate reward based on complexity reduction.
        
        Positive reward for reducing complexity.
        
        Args:
            original: Original metrics
            refined: Refined metrics
            normalize: Whether to normalize
            
        Returns:
            Complexity reward (positive = lower complexity)
        """
        complexity_reduction = original.cyclomatic_complexity - refined.cyclomatic_complexity
        
        if normalize:
            # Normalize based on typical complexity range
            complexity_reduction = self._normalize(
                complexity_reduction,
                -self.config.complexity_max,  # Worst case
                self.config.complexity_max    # Best case
            )
        
        return complexity_reduction
    
    def _calculate_readability_reward(
        self,
        original: TestMetrics,
        refined: TestMetrics
    ) -> float:
        """
        Calculate reward based on readability improvements.
        
        Uses heuristics:
        - Shorter code is often more readable (up to a point)
        - Lower complexity correlates with readability
        
        Args:
            original: Original metrics
            refined: Refined metrics
            
        Returns:
            Readability reward
        """
        # Heuristic: penalize extreme length differences
        loc_ratio = refined.lines_of_code / max(original.lines_of_code, 1)
        
        # Optimal range: 0.7 to 1.0 of original length
        if 0.7 <= loc_ratio <= 1.0:
            length_score = 1.0
        elif loc_ratio < 0.7:
            # Too short might lose important details
            length_score = 0.5
        else:
            # Too long
            length_score = max(0, 1.0 - (loc_ratio - 1.0))
        
        # Combine with complexity
        complexity_score = self._normalize(
            original.cyclomatic_complexity - refined.cyclomatic_complexity,
            -10, 10
        )
        
        return (length_score + complexity_score) / 2.0
    
    def _calculate_coverage_reward(
        self,
        original: TestMetrics,
        refined: TestMetrics
    ) -> float:
        """
        Calculate reward based on coverage preservation/improvement.
        
        We want to maintain or improve coverage.
        
        Args:
            original: Original metrics
            refined: Refined metrics
            
        Returns:
            Coverage reward
        """
        # Use branch coverage as primary metric
        coverage_delta = refined.branch_coverage - original.branch_coverage
        
        # Normalize to [-1, 1] range
        # We're more lenient with small drops but reward improvements
        if coverage_delta >= 0:
            # Improvement: scale from [0, 100] to [0, 1]
            reward = min(coverage_delta / 10.0, 1.0)
        else:
            # Drop: penalize more severely
            # Allow up to 5% drop with minor penalty
            if coverage_delta >= -5.0:
                reward = coverage_delta / 10.0  # Small penalty
            else:
                reward = -1.0  # Severe penalty for large drops
        
        return reward
    
    def _calculate_mutation_reward(
        self,
        original: TestMetrics,
        refined: TestMetrics
    ) -> float:
        """
        Calculate reward based on mutation score preservation/improvement.
        
        Args:
            original: Original metrics
            refined: Refined metrics
            
        Returns:
            Mutation reward
        """
        # Similar to coverage
        mutation_delta = refined.mutation_score - original.mutation_score
        
        if mutation_delta >= 0:
            reward = min(mutation_delta / 10.0, 1.0)
        else:
            # Allow small drops
            if mutation_delta >= -5.0:
                reward = mutation_delta / 10.0
            else:
                reward = -1.0
        
        return reward
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize value to [0, 1] range.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value
        """
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def get_detailed_feedback(
        self,
        original_metrics: TestMetrics,
        refined_metrics: TestMetrics
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        
        Useful for analysis and debugging.
        
        Args:
            original_metrics: Original test metrics
            refined_metrics: Refined test metrics
            
        Returns:
            Dict with individual reward components
        """
        return {
            'total_reward': self.calculate_reward(original_metrics, refined_metrics),
            'smells_reward': self._calculate_smell_reward(original_metrics, refined_metrics, True),
            'complexity_reward': self._calculate_complexity_reward(original_metrics, refined_metrics, True),
            'readability_reward': self._calculate_readability_reward(original_metrics, refined_metrics),
            'coverage_reward': self._calculate_coverage_reward(original_metrics, refined_metrics),
            'mutation_reward': self._calculate_mutation_reward(original_metrics, refined_metrics),
            'compiles': refined_metrics.compiles,
            'passes': refined_metrics.passes,
            # Deltas for analysis
            'smell_delta': sum(original_metrics.test_smells.values()) - sum(refined_metrics.test_smells.values()),
            'complexity_delta': original_metrics.cyclomatic_complexity - refined_metrics.cyclomatic_complexity,
            'coverage_delta': refined_metrics.branch_coverage - original_metrics.branch_coverage,
            'mutation_delta': refined_metrics.mutation_score - original_metrics.mutation_score
        }
