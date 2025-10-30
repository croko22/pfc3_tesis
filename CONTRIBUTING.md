# Contributing to GSPO-UTG

Thank you for your interest in contributing to the GSPO-based Unit Test Generation project!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/gspo_utg_tesis.git
   cd gspo_utg_tesis
   ```
3. **Set up development environment**:
   ```bash
   ./setup.sh
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow these guidelines:

- **Code Style**: Follow PEP 8
  ```bash
  black src/ experiments/
  flake8 src/ experiments/
  ```

- **Type Hints**: Use type hints where appropriate
  ```python
  def process_test(test_code: str, metrics: TestMetrics) -> float:
      ...
  ```

- **Documentation**: Add docstrings to functions and classes
  ```python
  def my_function(param: str) -> int:
      """
      Brief description.
      
      Args:
          param: Description of param
          
      Returns:
          Description of return value
      """
  ```

### 3. Test Your Changes

```bash
# Run examples to test basic functionality
python examples.py

# Run validation
python validate_setup.py

# Run tests if available
pytest tests/
```

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description

Detailed explanation of changes if needed.
Fixes #issue_number"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Areas for Contribution

### High Priority

- **Test Coverage**: Add unit tests for core components
- **Documentation**: Improve inline documentation and examples
- **Performance**: Optimize GSPO training loop
- **Benchmarks**: Add more benchmark datasets beyond Defects4J

### Medium Priority

- **Visualizations**: Add training progress visualizations
- **Experiment Tracking**: Better integration with W&B/MLflow
- **Model Support**: Test with more LLM architectures
- **Metrics**: Add more test quality metrics

### Low Priority

- **UI/Dashboard**: Web interface for experiment monitoring
- **Parallelization**: Multi-GPU training support
- **Caching**: Cache intermediate results

## Code Organization

```
src/
├── benchmark_handler/   # Dataset loading and management
├── evaluation/          # Test quality metrics
├── gspo_optimizer/      # GSPO algorithm implementation
├── llm_agent/          # LLM with LoRA
├── rl_env/             # RL environment and reward
└── static_analyzer/    # Java code analysis
```

## Adding New Features

### Adding a New Metric

1. Add to `src/evaluation/metrics.py`:
   ```python
   def calculate_new_metric(test_code: str) -> float:
       """Calculate new metric."""
       ...
   ```

2. Update `TestMetrics` dataclass
3. Integrate into reward function if needed
4. Update config.yml with default parameters

### Adding a New Test Smell Detector

1. Add to `JavaAnalyzer.detect_test_smells()` in `src/static_analyzer/extractor.py`
2. Add smell name to config.yml
3. Update documentation

### Adding Support for New LLM

1. Update `src/llm_agent/agent.py`
2. Add model-specific configuration to config.yml
3. Test with validation script

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- **Code**: Use Google-style docstrings
- **README**: Update for new features
- **Examples**: Add examples for new functionality

## Questions?

- Open an issue for bugs or feature requests
- Discussions for general questions
- Email [your-email@example.com] for other inquiries

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
