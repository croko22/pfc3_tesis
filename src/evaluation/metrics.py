"""
Evaluation metrics for unit tests.
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
# from ..static_analyzer.extractor import JavaAnalyzer
import re

logger = logging.getLogger(__name__)

@dataclass
class TestMetrics:
    """Container for all test metrics."""
    # Effectiveness metrics
    branch_coverage: float = 0.0
    line_coverage: float = 0.0
    method_coverage: float = 0.0
    mutation_score: float = 0.0
    
    # Maintainability metrics
    test_smells: Dict[str, int] = None
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    lines_of_code: int = 0
    
    # Compilation and execution
    compiles: bool = False
    passes: bool = False
    
    def __post_init__(self):
        if self.test_smells is None:
            self.test_smells = {}

class TestQualityEvaluator:
    """Evaluator using standard Java tools."""
    
    def __init__(self, project_dir: Path, classpath: str):
        self.project_dir = project_dir
        self.classpath = classpath
        self.junit_jar = "lib/junit-4.11.jar" # Assumed location
        self.hamcrest_jar = "lib/hamcrest-core-1.3.jar" # Assumed location
        self.analyzer = None # JavaAnalyzer()
        
    def evaluate_test(self, test_code: str, test_class_name: str) -> TestMetrics:
        metrics = TestMetrics()
        
        # 1. Compile
        compiles, test_file = self._compile_test(test_code, test_class_name)
        metrics.compiles = compiles
        
        if not compiles:
            return metrics
            
        # 2. Run
        metrics.passes = self._run_test(test_class_name)
        
        # code_metrics = self.analyzer.get_code_summary(test_code)
        # metrics.test_smells = self.analyzer.detect_test_smells(test_code)
        metrics.test_smells = {}
        
        # TODO: Implement coverage and static analysis
        
        return metrics
        
    def _compile_test(self, test_code: str, test_class_name: str) -> Tuple[bool, Optional[Path]]:
        # Write test file
        test_dir = self.project_dir / "temp_tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_dir / f"{test_class_name}.java"
        with open(test_file, 'w') as f:
            f.write(test_code)
            
        # Compile
        cp = f"{self.classpath}:{self.junit_jar}:{self.hamcrest_jar}"
        cmd = ["javac", "-cp", cp, str(test_file)]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True, test_file
        except subprocess.CalledProcessError:
            return False, None

    def _run_test(self, test_class_name: str) -> bool:
        cp = f"{self.classpath}:{self.junit_jar}:{self.hamcrest_jar}:{self.project_dir}/temp_tests"
        cmd = ["java", "-cp", cp, "org.junit.runner.JUnitCore", test_class_name]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return "OK (" in result.stdout
        except Exception:
            return False
