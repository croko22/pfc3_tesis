"""
Main experiment runner for GSPO-based Unit Test Generation.

This script orchestrates the complete pipeline:
1. Load Defects4J benchmark
2. Generate initial tests with EvoSuite
3. Refine tests with GSPO-optimized LLM
4. Evaluate and compare results
"""

import argparse
import logging
import yaml
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_handler.loader import Defects4JLoader
from src.static_analyzer.extractor import JavaAnalyzer
from src.evaluation.metrics import TestQualityEvaluator, TestMetrics
from src.rl_env.reward import RewardFunction, RewardConfig
from src.rl_env.environment import TestRefinementEnvironment
from src.llm_agent.agent import LLMAgent
from src.gspo_optimizer.optimizer import GSPOOptimizer, GSPOConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvoSuiteGenerator:
    """Wrapper for EvoSuite test generation."""
    
    def __init__(self, evosuite_jar: Path, config: Dict):
        """
        Initialize EvoSuite generator.
        
        Args:
            evosuite_jar: Path to EvoSuite JAR
            config: EvoSuite configuration
        """
        self.evosuite_jar = evosuite_jar
        self.config = config
    
    def generate_tests(
        self,
        project_dir: Path,
        class_name: str,
        output_dir: Path
    ) -> Optional[Path]:
        """
        Generate tests for a class using EvoSuite.
        
        Args:
            project_dir: Project directory
            class_name: Fully qualified class name
            output_dir: Output directory for tests
            
        Returns:
            Path to generated test file or None if failed
        """
        logger.info(f"Generating tests for {class_name} with EvoSuite")
        
        # Build EvoSuite command
        cmd = [
            "java",
            "-jar", str(self.evosuite_jar),
            "-class", class_name,
            "-projectCP", str(project_dir / "target" / "classes"),
            "-Dsearch_budget", str(self.config.get("search_budget", 120)),
            "-Dcriterion", self.config.get("criterion", "BRANCH"),
            "-Dassertions", str(self.config.get("assertions", True)).lower(),
            "-Dminimize", str(self.config.get("minimize", True)).lower(),
        ]
        
        # Add extra parameters
        extra_params = self.config.get("extra_params", "")
        if extra_params:
            cmd.extend(extra_params.split())
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=self.config.get("search_budget", 120) + 60
            )
            
            if result.returncode != 0:
                logger.error(f"EvoSuite failed: {result.stderr}")
                return None
            
            # Find generated test file
            evosuite_tests = project_dir / "evosuite-tests"
            test_file = self._find_test_file(evosuite_tests, class_name)
            
            if test_file:
                logger.info(f"Generated test: {test_file}")
                return test_file
            else:
                logger.warning(f"No test file found for {class_name}")
                return None
        
        except subprocess.TimeoutExpired:
            logger.error(f"EvoSuite timed out for {class_name}")
            return None
        except Exception as e:
            logger.error(f"Error running EvoSuite: {e}")
            return None
    
    def _find_test_file(self, search_dir: Path, class_name: str) -> Optional[Path]:
        """Find generated test file."""
        # Convert class name to path
        simple_name = class_name.split('.')[-1]
        test_name = f"{simple_name}_ESTest.java"
        
        for test_file in search_dir.rglob(test_name):
            return test_file
        
        return None


class ExperimentRunner:
    """Main experiment runner."""
    
    def __init__(self, config_path: Path):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.workspace = Path(self.config['paths']['workspace'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.checkpoints_dir = Path(self.config['paths']['checkpoints_dir'])
        
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing experiment components...")
        
        # Defects4J loader
        self.defects4j_loader = Defects4JLoader(
            self.config['paths']['defects4j_home'],
            str(self.workspace)
        )
        
        # EvoSuite generator
        self.evosuite = EvoSuiteGenerator(
            Path(self.config['paths']['evosuite_jar']),
            self.config['evosuite']
        )
        
        # Static analyzer
        self.analyzer = JavaAnalyzer()
        
        # Reward function
        reward_config = RewardConfig(**self.config['reward']['weights'])
        reward_config.failure_penalty = self.config['reward']['failure_penalty']
        self.reward_function = RewardFunction(reward_config)
        
        # LLM agent (will be initialized during training)
        self.agent: Optional[LLMAgent] = None
        self.optimizer: Optional[GSPOOptimizer] = None
        
        logger.info("Initialization complete")
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING FULL EXPERIMENT")
        logger.info("=" * 80)
        
        # 1. Load benchmark data
        logger.info("\n[1/5] Loading Defects4J benchmark...")
        benchmark_data = self.load_benchmark()
        
        # 2. Generate initial tests with EvoSuite
        logger.info("\n[2/5] Generating tests with EvoSuite...")
        evosuite_tests = self.generate_evosuite_tests(benchmark_data)
        
        # 3. Initialize and train LLM with GSPO
        logger.info("\n[3/5] Training LLM with GSPO...")
        self.train_llm(evosuite_tests)
        
        # 4. Refine tests with trained LLM
        logger.info("\n[4/5] Refining tests with GSPO-optimized LLM...")
        refined_tests = self.refine_tests(evosuite_tests)
        
        # 5. Evaluate and compare
        logger.info("\n[5/5] Evaluating and comparing results...")
        results = self.evaluate_results(evosuite_tests, refined_tests)
        
        # Save results
        self.save_results(results)
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        
        return results
    
    def load_benchmark(self) -> List[Dict]:
        """Load Defects4J benchmark data."""
        projects = self.config['defects4j']['projects']
        bug_ids = self.config['defects4j'].get('bug_ids', [])
        max_bugs = self.config['defects4j'].get('max_bugs_per_project', 5)
        
        benchmark_data = []
        
        for bug_info in self.defects4j_loader.load_benchmark(projects, bug_ids, max_bugs):
            # Extract modified classes
            for class_name in bug_info['modified_classes']:
                # Get class source file
                class_path = self.defects4j_loader.get_class_path(
                    bug_info['buggy_dir'],
                    class_name
                )
                
                if class_path and class_path.exists():
                    with open(class_path, 'r') as f:
                        source_code = f.read()
                    
                    benchmark_data.append({
                        'project': bug_info['project'],
                        'bug_id': bug_info['bug_id'],
                        'class_name': class_name,
                        'source_code': source_code,
                        'project_dir': bug_info['buggy_dir']
                    })
        
        logger.info(f"Loaded {len(benchmark_data)} classes from benchmark")
        return benchmark_data
    
    def generate_evosuite_tests(self, benchmark_data: List[Dict]) -> List[Dict]:
        """Generate tests using EvoSuite."""
        evosuite_tests = []
        
        for idx, item in enumerate(benchmark_data):
            logger.info(
                f"[{idx + 1}/{len(benchmark_data)}] Generating test for "
                f"{item['project']}-{item['bug_id']}: {item['class_name']}"
            )
            
            # Generate test
            test_file = self.evosuite.generate_tests(
                item['project_dir'],
                item['class_name'],
                self.workspace / "evosuite_tests"
            )
            
            if test_file:
                with open(test_file, 'r') as f:
                    test_code = f.read()
                
                evosuite_tests.append({
                    **item,
                    'evosuite_test_code': test_code,
                    'evosuite_test_file': test_file
                })
            else:
                logger.warning(f"Failed to generate test for {item['class_name']}")
        
        logger.info(f"Generated {len(evosuite_tests)} tests with EvoSuite")
        return evosuite_tests
    
    def train_llm(self, training_data: List[Dict]):
        """Train LLM using GSPO."""
        # Initialize LLM agent
        logger.info("Initializing LLM agent...")
        self.agent = LLMAgent(
            model_name=self.config['llm']['model_name'],
            use_lora=self.config['llm']['use_lora'],
            lora_config=self.config['llm']['lora'],
            generation_config=self.config['llm']['generation'],
            device=self.config['llm']['device']
        )
        
        # Initialize GSPO optimizer
        logger.info("Initializing GSPO optimizer...")
        gspo_config = GSPOConfig(**self.config['gspo'])
        self.optimizer = GSPOOptimizer(
            self.agent,
            gspo_config,
            self.checkpoints_dir
        )
        
        # Prepare training dataset
        logger.info("Preparing training dataset...")
        train_dataset = self._prepare_training_data(training_data)
        
        # Split into train/val
        split_idx = int(len(train_dataset) * self.config['experiment']['train_split'])
        train_data = train_dataset[:split_idx]
        val_data = train_dataset[split_idx:]
        
        logger.info(f"Training on {len(train_data)} examples, validating on {len(val_data)}")
        
        # Train
        history = self.optimizer.train(train_data, val_data)
        
        logger.info("Training complete!")
        return history
    
    def _prepare_training_data(self, evosuite_tests: List[Dict]) -> List[Dict]:
        """
        Prepare training data by generating initial refinements and computing rewards.
        """
        training_data = []
        
        for item in evosuite_tests:
            # Create environment
            evaluator = TestQualityEvaluator(
                item['project_dir'],
                Path(self.config['paths'].get('pit_jar'))
            )
            
            env = TestRefinementEnvironment(
                item['project_dir'],
                evaluator,
                self.reward_function,
                self.analyzer
            )
            
            # Reset environment
            state = env.reset(
                item['evosuite_test_code'],
                item['source_code'],
                item['class_name']
            )
            
            # Get prompt
            prompt = state.to_prompt()
            
            # For initial training, we'll use a simple approach:
            # Generate a refinement and compute its reward
            if self.agent:
                refined_code = self.agent.generate([prompt])[0]
            else:
                # Skip if agent not initialized
                continue
            
            # Evaluate
            _, reward, _, info = env.step(refined_code)
            
            training_data.append({
                'prompt': prompt,
                'response': refined_code,
                'reward': reward,
                'info': info
            })
        
        return training_data
    
    def refine_tests(self, evosuite_tests: List[Dict]) -> List[Dict]:
        """Refine tests using trained LLM."""
        refined_tests = []
        
        for idx, item in enumerate(evosuite_tests):
            logger.info(
                f"[{idx + 1}/{len(evosuite_tests)}] Refining test for "
                f"{item['project']}-{item['bug_id']}: {item['class_name']}"
            )
            
            # Create environment
            evaluator = TestQualityEvaluator(
                item['project_dir'],
                Path(self.config['paths'].get('pit_jar'))
            )
            
            env = TestRefinementEnvironment(
                item['project_dir'],
                evaluator,
                self.reward_function,
                self.analyzer
            )
            
            # Reset
            state = env.reset(
                item['evosuite_test_code'],
                item['source_code'],
                item['class_name']
            )
            
            # Generate refinement
            prompt = state.to_prompt()
            refined_code = self.agent.generate([prompt])[0]
            
            # Evaluate
            _, reward, _, info = env.step(refined_code)
            
            refined_tests.append({
                **item,
                'refined_test_code': refined_code,
                'reward': reward,
                'info': info
            })
        
        return refined_tests
    
    def evaluate_results(
        self,
        evosuite_tests: List[Dict],
        refined_tests: List[Dict]
    ) -> Dict:
        """Evaluate and compare results."""
        results = {
            'evosuite': [],
            'refined_gspo': [],
            'comparison': {}
        }
        
        # Collect metrics
        for evosuite, refined in zip(evosuite_tests, refined_tests):
            results['evosuite'].append({
                'class': evosuite['class_name'],
                'metrics': refined['info']['original_metrics']
            })
            
            results['refined_gspo'].append({
                'class': refined['class_name'],
                'metrics': refined['info']['refined_metrics'],
                'reward': refined['reward']
            })
        
        # Compute aggregate statistics
        results['comparison'] = self._compute_comparison_stats(
            results['evosuite'],
            results['refined_gspo']
        )
        
        return results
    
    def _compute_comparison_stats(
        self,
        evosuite_results: List[Dict],
        refined_results: List[Dict]
    ) -> Dict:
        """Compute comparison statistics."""
        stats = {}
        
        # Extract metrics
        metrics_names = [
            'branch_coverage',
            'mutation_score',
            'total_smells',
            'cyclomatic_complexity'
        ]
        
        for metric in metrics_names:
            evosuite_values = [r['metrics'].get(metric, 0) for r in evosuite_results]
            refined_values = [r['metrics'].get(metric, 0) for r in refined_results]
            
            stats[metric] = {
                'evosuite_mean': np.mean(evosuite_values),
                'evosuite_std': np.std(evosuite_values),
                'refined_mean': np.mean(refined_values),
                'refined_std': np.std(refined_values),
                'improvement': np.mean(refined_values) - np.mean(evosuite_values)
            }
        
        # Overall metrics
        stats['overall'] = {
            'evosuite_pass_rate': np.mean([r['metrics'].get('passes', False) for r in evosuite_results]),
            'refined_pass_rate': np.mean([r['metrics'].get('passes', False) for r in refined_results]),
            'avg_reward': np.mean([r.get('reward', 0) for r in refined_results])
        }
        
        return stats
    
    def save_results(self, results: Dict):
        """Save experimental results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"results_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_results = self._to_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Also save a summary
        summary_file = self.output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(json.dumps(results['comparison'], indent=2))
        
        logger.info(f"Summary saved to {summary_file}")
    
    def _to_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_json_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GSPO-based Unit Test Generation Experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "generate", "train", "refine", "evaluate"],
        default="full",
        help="Experiment mode"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner(args.config)
    
    # Run experiment
    if args.mode == "full":
        runner.run_full_experiment()
    else:
        logger.warning(f"Mode '{args.mode}' not fully implemented yet")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
