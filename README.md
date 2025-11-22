# 🎓 Test Generation + LLM Refinement# GSPO-based Unit Test Generation and Refinement



Metodología completa de investigación para mejora de tests automáticos mediante LLM.[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

## 🚀 Quick Start

This project implements a novel hybrid approach to automatic unit test generation that combines the strengths of traditional search-based methods (EvoSuite) with modern large language models (LLMs) optimized using Group Sequence Policy Optimization (GSPO).

```bash

# 1. Ver estado del proyecto### Key Features

python scripts/setup/status.py

- **Hybrid Test Generation**: EvoSuite generates initial high-coverage tests, LLM refines them for maintainability

# 2. Test rápido (1 clase, 60s)- **GSPO Optimization**: Stable reinforcement learning training using sequence-level importance sampling

python scripts/testing/quick_test.py- **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation

- **Multi-Objective Reward**: Balances test effectiveness (coverage, mutation score) and maintainability (smells, complexity)

# 3. Pipeline completo (10 clases, 30 min)- **Defects4J Benchmark**: Comprehensive evaluation on real-world bugs

python scripts/pipeline/run_pipeline.py --limit 10

```## Architecture



---```

┌─────────────────┐

## 📁 Estructura│  Defects4J      │

│  Benchmark      │

```└────────┬────────┘

scripts/         │

  pipeline/      → Pipeline completo de 5 fases         v

  testing/       → Tests rápidos y validación┌─────────────────┐

  setup/         → Setup y verificación│  EvoSuite       │

│  Test Generator │

docs/└────────┬────────┘

  guides/        → Guías principales (LEE PRIMERO)         │ Initial Tests (High Coverage)

  legacy/        → Docs/código viejo (ignorar)         v

┌─────────────────┐

data/            → Benchmarks (SF110 + Extended DynaMOSA)│  LLM Refiner    │

lib/             → Librerías Java (EvoSuite, JaCoCo, JUnit)│  (GSPO + LoRA)  │

└────────┬────────┘

baseline_tests/  → Outputs del pipeline         │ Refined Tests (High Maintainability)

refined_tests/         v

valid_tests/┌─────────────────┐

evaluation_results/│  Evaluator      │

figures/│  (Metrics)      │

```└─────────────────┘

```

---

## Installation

## 📖 Documentación

### Prerequisites

**Lee en orden**:

1. `docs/guides/START_HERE.md` - Guía rápida- Python 3.8 or higher

2. `docs/guides/CHECKLIST.md` - TODOs pendientes- Java 8 or higher (for EvoSuite and Defects4J)

3. `docs/guides/METHODOLOGY.md` - Metodología completa- CUDA-capable GPU (recommended) or CPU

- Git

**Resumen**: `README_ORGANIZED.md`

### 1. Clone the Repository

---

```bash

## 🧹 Limpiezagit clone https://github.com/yourusername/gspo_utg_tesis.git

cd gspo_utg_tesis

```bash```

# Ver qué está organizado

cat README_ORGANIZED.md### 2. Install Python Dependencies



# Limpiar archivos viejos/innecesarios```bash

./cleanup.sh# Create virtual environment

```python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

---

# Install dependencies

## 🎯 Scripts Principalespip install -r requirements.txt

```

| Script | Uso | Tiempo |

|--------|-----|--------|### 3. Install External Tools

| `scripts/setup/status.py` | Ver estado | 1s |

| `scripts/testing/quick_test.py` | Test 1 clase | 1 min |#### Defects4J

| `scripts/pipeline/run_pipeline.py` | Pipeline completo | Variable |

```bash

---# Clone and setup Defects4J

git clone https://github.com/rjust/defects4j.git /path/to/defects4j

## ⚠️ TODOs Pendientescd /path/to/defects4j

./init.sh

1. Conectar LLM en `phase2_llm_refinement.py`

2. Implementar PIT en `phase4_evaluation.py`# Add to PATH

3. Implementar JaCoCo en `phase4_evaluation.py`export PATH="/path/to/defects4j/framework/bin:$PATH"

```

Ver: `docs/guides/CHECKLIST.md`

#### EvoSuite

---

```bash

## 📊 Pipeline de 5 Fases# Download EvoSuite

wget https://github.com/EvoSuite/evosuite/releases/download/v1.2.0/evosuite-master-1.2.0.jar

```mv evosuite-master-1.2.0.jar /path/to/evosuite/

T_base → T_refined → T_valid → Métricas → Gráficas```

 (1)       (2)        (3)        (4)        (5)

```#### PIT (Mutation Testing - Optional)



**Estado**:```bash

- ✅ Fase 1, 3, 5: COMPLETAS# Download PIT

- ⚠️ Fase 2, 4: TEMPLATES (implementar TODOs)wget https://github.com/hcoles/pitest/releases/download/pitest-parent-1.14.2/pitest-command-line-1.14.2.jar

mv pitest-command-line-1.14.2.jar /path/to/pit/

---```



**README viejo**: `README_OLD.md`  ### 4. Configure the Project

**Fecha**: 2024-11-16  

**Versión**: 1.0 (Organizada)Edit `config.yml` to set your paths:


```yaml
paths:
  defects4j_home: "/path/to/defects4j"
  evosuite_jar: "/path/to/evosuite-master-1.2.0.jar"
  pit_jar: "/path/to/pitest-command-line-1.14.2.jar"
  workspace: "./workspace"
  output_dir: "./results"
  checkpoints_dir: "./checkpoints"
```

## Quick Start

### Run Full Experiment

```bash
python experiments/run_experiment.py --config config.yml --mode full
```

This will:
1. Load Defects4J projects
2. Generate initial tests with EvoSuite
3. Train LLM with GSPO
4. Refine tests
5. Evaluate and compare results

### Run Individual Steps

```bash
# Generate tests with EvoSuite only
python experiments/run_experiment.py --mode generate

# Train LLM
python experiments/run_experiment.py --mode train

# Refine existing tests
python experiments/run_experiment.py --mode refine

# Evaluate results
python experiments/run_experiment.py --mode evaluate
```

## Configuration

### Key Configuration Options

#### LLM Settings

```yaml
llm:
  model_name: "codellama/CodeLlama-7b-Instruct-hf"
  use_lora: true
  lora:
    r: 16  # Rank of LoRA matrices
    lora_alpha: 32
    lora_dropout: 0.05
  device: "cuda"  # or "cpu"
```

#### GSPO Settings

```yaml
gspo:
  learning_rate: 1.0e-5
  num_epochs: 3
  batch_size: 8
  clip_ratio: 0.2
  kl_coef: 0.1
```

#### Reward Function

```yaml
reward:
  weights:
    test_smells: -0.3
    cyclomatic_complexity: -0.2
    branch_coverage: 0.25
    mutation_score: 0.25
```

## Project Structure

```
gspo_utg_tesis/
├── config.yml                 # Main configuration
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── experiments/
│   └── run_experiment.py     # Main experiment runner
└── src/
    ├── benchmark_handler/
    │   └── loader.py         # Defects4J loader
    ├── evaluation/
    │   └── metrics.py        # Test quality metrics
    ├── gspo_optimizer/
    │   └── optimizer.py      # GSPO implementation
    ├── llm_agent/
    │   └── agent.py          # LLM with LoRA
    ├── rl_env/
    │   ├── environment.py    # RL environment
    │   └── reward.py         # Reward function
    └── static_analyzer/
        └── extractor.py      # Java code analyzer
```

## Research Questions (RQs)

### RQ1: Maintainability
**Do GSPO-refined tests have significantly fewer test smells and lower complexity than EvoSuite-generated tests?**

Metrics:
- Test smells frequency
- Cyclomatic complexity
- Lines of code

### RQ2: Effectiveness
**Do GSPO-refined tests maintain or improve effectiveness compared to EvoSuite-generated tests?**

Metrics:
- Branch coverage (%)
- Mutation score (%)
- Test pass rate

## Key Components

### 1. GSPO Optimizer

Implements the Group Sequence Policy Optimization algorithm:

```python
# Sequence-level importance sampling ratio
s_i(θ) = (π_θ(y|x) / π_θ_old(y|x))^(1/|y|)
```

This provides more stable training than PPO's token-level ratios.

### 2. Multi-Objective Reward Function

Balances multiple objectives:

```python
R = w_smells · r_smells + 
    w_complexity · r_complexity + 
    w_coverage · r_coverage + 
    w_mutation · r_mutation
```

### 3. Test Quality Evaluator

Comprehensively evaluates tests on:
- **Effectiveness**: Coverage (branch, line, method), mutation score
- **Maintainability**: Test smells, complexity metrics
- **Correctness**: Compilation, execution success

## Example Usage

### Training a Model

```python
from src.llm_agent.agent import LLMAgent
from src.gspo_optimizer.optimizer import GSPOOptimizer, GSPOConfig

# Initialize agent
agent = LLMAgent(
    model_name="codellama/CodeLlama-7b-Instruct-hf",
    use_lora=True
)

# Initialize optimizer
config = GSPOConfig(learning_rate=1e-5, num_epochs=3)
optimizer = GSPOOptimizer(agent, config, output_dir="./checkpoints")

# Train
history = optimizer.train(train_dataset, val_dataset)
```

### Refining a Test

```python
from src.rl_env.environment import TestRefinementEnvironment

# Create environment
env = TestRefinementEnvironment(
    project_dir=Path("./project"),
    evaluator=evaluator,
    reward_function=reward_fn
)

# Reset with EvoSuite test
state = env.reset(
    original_test_code=evosuite_test,
    source_code=source_code,
    source_class_name="MyClass"
)

# Generate refined test
prompt = state.to_prompt()
refined_test = agent.generate([prompt])[0]

# Evaluate
state, reward, done, info = env.step(refined_test)
```

## Results

Results are saved in the `results/` directory:

- `results_YYYYMMDD_HHMMSS.json`: Full detailed results
- `summary_YYYYMMDD_HHMMSS.txt`: Summary statistics

### Interpreting Results

The comparison statistics include:

```json
{
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
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

Solution: Reduce batch size or use 4-bit quantization:

```yaml
llm:
  generation:
    batch_size: 2
```

**2. EvoSuite Timeout**

Solution: Increase search budget:

```yaml
evosuite:
  search_budget: 180  # Increase from 120
```

**3. Compilation Errors**

Solution: Ensure Defects4J project is properly checked out and compiled:

```bash
defects4j checkout -p Lang -v 1b -w /tmp/lang-1b
cd /tmp/lang-1b
defects4j compile
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{yourthesis2025,
  title={GSPO-based Unit Test Generation and Refinement},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## References

- [EvoSuite](https://www.evosuite.org/) - Fraser and Arcuri, 2011
- [Defects4J](https://github.com/rjust/defects4j) - Just et al., 2014
- [GSPO](https://arxiv.org/abs/2310.11346) - Zheng et al., 2023
- [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., 2021

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EvoSuite team for the excellent search-based test generation tool
- Defects4J maintainers for the comprehensive bug benchmark
- HuggingFace for transformers and PEFT libraries
- Authors of GSPO for the stable RL optimization algorithm

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]
