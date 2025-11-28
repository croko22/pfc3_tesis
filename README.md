
## 🚀 Usage

The project uses a modular pipeline located in `scripts/pipeline/run_pipeline.py`.

### Run Full Pipeline
```bash
python3 scripts/pipeline/run_pipeline.py --full
```

### Run Specific Phase
```bash
# Phase 1: Baseline Generation (EvoSuite)
python3 scripts/pipeline/run_pipeline.py --phase 1

# Phase 2: LLM Refinement
python3 scripts/pipeline/run_pipeline.py --phase 2

# Phase 3: Verification
python3 scripts/pipeline/run_pipeline.py --phase 3

# Phase 4: Evaluation
python3 scripts/pipeline/run_pipeline.py --phase 4
```

### Quick Test
Run with a limit to test the pipeline on a small subset of classes:
```bash
python3 scripts/pipeline/run_pipeline.py --limit 5
```
