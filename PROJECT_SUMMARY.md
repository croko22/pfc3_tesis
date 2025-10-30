# GSPO-UTG Project Summary

## Overview

This project implements a **hybrid approach to automatic unit test generation** that combines:

1. **EvoSuite** - Search-based test generation for high code coverage
2. **LLM with GSPO** - Reinforcement learning-based test refinement for high maintainability
3. **LoRA** - Efficient fine-tuning with minimal computational cost

## Key Innovation: GSPO for Stable Training

### The Problem

Traditional RL approaches (like PPO) for LLM optimization suffer from **training instability** when applied to code generation tasks. Token-level importance sampling ratios cause high variance in gradients.

### The Solution: Group Sequence Policy Optimization

GSPO uses **sequence-level** importance sampling ratios instead:

```
s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
```

This normalization by sequence length `|y_i|` significantly reduces gradient variance, enabling stable training.

## Architecture Components

### 1. Benchmark Handler (`src/benchmark_handler/`)
- **Defects4JLoader**: Loads real-world bugs from Defects4J
- Supports multiple projects: Chart, Closure, Lang, Math, Time
- Handles project checkout, compilation, and metadata extraction

### 2. Static Analyzer (`src/static_analyzer/`)
- **JavaAnalyzer**: AST-based Java code analysis using javalang
- Extracts methods, complexity metrics, code structure
- Detects 10+ test smells (Assertion Roulette, Eager Test, etc.)
- Computes cyclomatic complexity and LOC

### 3. Evaluation Metrics (`src/evaluation/`)
- **TestQualityEvaluator**: Comprehensive test assessment
  - **Effectiveness**: Branch/line/method coverage, mutation score (PIT)
  - **Maintainability**: Test smells, complexity, readability
  - **Correctness**: Compilation and execution success

### 4. RL Environment (`src/rl_env/`)
- **TestRefinementEnvironment**: RL wrapper for test refinement
  - State: Original test + source code + context
  - Action: Refined test code
  - Reward: Multi-objective quality score
- **RewardFunction**: Balances 6 components with configurable weights

### 5. LLM Agent (`src/llm_agent/`)
- **LLMAgent**: LLM with LoRA for efficient fine-tuning
- Supports any HuggingFace model (CodeLlama, StarCoder, etc.)
- 4-bit/8-bit quantization for memory efficiency
- Generates refined tests from prompts

### 6. GSPO Optimizer (`src/gspo_optimizer/`)
- **GSPOOptimizer**: Implements GSPO algorithm
- Features:
  - Sequence-level policy ratios
  - Gradient clipping and normalization
  - Warmup scheduling
  - Checkpoint management
  - Training history tracking

### 7. Experiment Runner (`experiments/`)
- **run_experiment.py**: Orchestrates full pipeline
- Modes: full, generate, train, refine, evaluate
- Automatic result saving and analysis

## Workflow

### Phase 1: Test Generation (EvoSuite)
```
Source Code → EvoSuite → Initial Tests (High Coverage, Low Maintainability)
```

### Phase 2: Model Training (GSPO)
```
Initial Tests → LLM Agent → Refined Tests → Reward Calculation → GSPO Update
```

### Phase 3: Test Refinement (Trained LLM)
```
Initial Tests → Trained LLM → Final Refined Tests (High Coverage + High Maintainability)
```

### Phase 4: Evaluation
```
Compare: EvoSuite vs GSPO-Refined vs Baseline-LLM
Metrics: Coverage, Mutation Score, Test Smells, Complexity
```

## Research Questions

### RQ1: Maintainability
**Do GSPO-refined tests have significantly fewer test smells and lower complexity than EvoSuite-generated tests?**

**Hypothesis**: Yes, the multi-objective reward function and GSPO optimization should reduce test smells by ~50% and complexity by ~30%.

### RQ2: Effectiveness
**Do GSPO-refined tests maintain or improve effectiveness (coverage, mutation score) compared to EvoSuite-generated tests?**

**Hypothesis**: Yes, the reward function penalizes effectiveness loss, so refined tests should maintain ≥95% of original coverage while improving maintainability.

## Configuration

All parameters in `config.yml`:

```yaml
# EvoSuite: Search budget, coverage criterion
evosuite:
  search_budget: 120
  criterion: "BRANCH"

# LLM: Model selection, LoRA config
llm:
  model_name: "codellama/CodeLlama-7b-Instruct-hf"
  lora:
    r: 16
    lora_alpha: 32

# GSPO: Learning rate, clipping, KL penalty
gspo:
  learning_rate: 1.0e-5
  clip_ratio: 0.2
  kl_coef: 0.1

# Reward: Component weights
reward:
  weights:
    test_smells: -0.3
    branch_coverage: 0.25
    mutation_score: 0.25
```

## Usage

### Quick Start
```bash
# Setup
./setup.sh
source venv/bin/activate

# Validate
python validate_setup.py

# Run examples
python examples.py

# Run full experiment
python experiments/run_experiment.py --config config.yml
```

### Custom Experiments
```python
from src.llm_agent.agent import LLMAgent
from src.gspo_optimizer.optimizer import GSPOOptimizer

# Initialize
agent = LLMAgent(model_name="codellama/CodeLlama-7b-Instruct-hf")
optimizer = GSPOOptimizer(agent, config, output_dir="./checkpoints")

# Train
history = optimizer.train(train_data, val_data)

# Refine
refined_code = agent.generate([prompt])[0]
```

## Results Format

```json
{
  "comparison": {
    "branch_coverage": {
      "evosuite_mean": 75.5,
      "refined_mean": 76.2,
      "improvement": 0.7
    },
    "total_smells": {
      "evosuite_mean": 4.2,
      "refined_mean": 1.8,
      "improvement": -2.4
    }
  }
}
```

## Dependencies

### Core
- PyTorch 2.0+
- Transformers 4.35+
- PEFT 0.6+
- javalang 0.13+

### External Tools
- Defects4J
- EvoSuite 1.2.0+
- PIT (optional)
- Java 8+

## Future Work

1. **Multi-turn Refinement**: Iterative refinement with feedback
2. **Self-Consistency**: Generate multiple candidates and select best
3. **Additional Benchmarks**: SF110, GitHub projects
4. **Alternative LLMs**: GPT-4, Claude, Gemini
5. **Active Learning**: Select most informative examples for training

## References

- **GSPO**: Zheng et al., "Group Preference Optimization", 2023
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
- **EvoSuite**: Fraser & Arcuri, "EvoSuite: Automatic Test Suite Generation", 2011
- **Defects4J**: Just et al., "Defects4J: A Database of Existing Faults", 2014

## Project Statistics

- **Lines of Code**: ~3,500
- **Modules**: 7 main components
- **Configuration Options**: 50+
- **Supported Test Smells**: 10
- **Metrics Tracked**: 15+

## Contact

For questions, issues, or contributions:
- GitHub Issues: [project-url]/issues
- Email: [your-email]
- Documentation: See README.md

---

**Last Updated**: 2025-11-04
**Version**: 0.1.0
**Status**: Active Development
