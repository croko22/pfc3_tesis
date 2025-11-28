import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from ..core.config import cfg
from ..core.loader import loader
from ..core.runner import runner

class EvaluationPhase:
    """Phase 4: Evaluate metrics."""
    
    def run(self) -> Dict:
        valid_file = cfg.base_dir / "generated_tests/validated/T_valid_results.json"
        if not valid_file.exists():
            print("❌ T_valid not found")
            return {"success": False}
            
        with open(valid_file) as f:
            valid_results = json.load(f)
            
        verified = [r for r in valid_results if r.get('verified')]
        metrics = []
        
        print(f"Phase 4: Evaluating {len(verified)} tests")
        
        for i, item in enumerate(verified, 1):
            project_name = item['project']
            class_name = item['class']
            
            print(f"[{i}/{len(verified)}] {class_name}")
            
            project = loader.get_project(project_name)
            if not project: continue
            sut_jar = project.jar_files[0]
            
            refined_path = Path(item['refined_file'])
            
            # Measure
            m = self._measure_metrics(refined_path, sut_jar, class_name)
            metrics.append({
                "project": project_name,
                "class": class_name,
                **m
            })
            
        # Save
        out_file = cfg.base_dir / "evaluation_results/final_evaluation.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return {"success": True, "count": len(metrics)}
        
    def _measure_metrics(self, test_path: Path, sut_jar: Path, class_name: str) -> Dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            # Compile
            cp = f"{sut_jar}:{cfg.junit_jar}:{cfg.evosuite_jar}:{test_path.parent}"
            runner.run_javac(cp, [test_path], out_dir)
            
            test_class = class_name + "_ESTest"
            
            # Coverage (JaCoCo)
            # TODO: Implement full JaCoCo parsing logic here or in runner
            # For now returning placeholder to match previous logic structure
            # Real implementation would use runner.run_java with javaagent
            
            return {
                "line_coverage": 0.0, # Placeholder
                "mutation_score": 0.0 # Placeholder
            }
