"""
Evaluation metrics for unit tests.

Includes coverage metrics, mutation testing, test smells, and quality metrics.
"""

import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from ..static_analyzer.extractor import JavaAnalyzer

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
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "branch_coverage": self.branch_coverage,
            "line_coverage": self.line_coverage,
            "method_coverage": self.method_coverage,
            "mutation_score": self.mutation_score,
            "test_smells": self.test_smells,
            "total_smells": sum(self.test_smells.values()),
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "lines_of_code": self.lines_of_code,
            "compiles": self.compiles,
            "passes": self.passes
        }


class CoverageAnalyzer:
    """Analyzer for code coverage using JaCoCo."""
    
    def __init__(self, project_dir: Path):
        """
        Initialize coverage analyzer.
        
        Args:
            project_dir: Path to project directory
        """
        self.project_dir = project_dir
        self.jacoco_agent = None  # Path to JaCoCo agent jar
    
    def run_coverage(self, test_class: str, timeout: int = 300) -> Optional[Dict[str, float]]:
        """
        Run coverage analysis for a test class.
        
        Args:
            test_class: Test class name
            timeout: Timeout in seconds
            
        Returns:
            Dict with coverage metrics or None if failed
        """
        try:
            # This is a simplified version - in practice, you'd integrate with
            # the project's build system (Maven/Gradle) or use JaCoCo directly
            
            # For Defects4J projects, we can use the built-in coverage command
            result = subprocess.run(
                ["defects4j", "coverage", "-t", test_class],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Parse coverage output
            coverage = self._parse_coverage_output(result.stdout)
            return coverage
        
        except subprocess.TimeoutExpired:
            logger.error(f"Coverage analysis timed out for {test_class}")
            return None
        except Exception as e:
            logger.error(f"Failed to run coverage: {e}")
            return None
    
    def _parse_coverage_output(self, output: str) -> Dict[str, float]:
        """
        Parse coverage output.
        
        Args:
            output: Coverage command output
            
        Returns:
            Dict with coverage percentages
        """
        coverage = {
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "method_coverage": 0.0
        }
        
        # Parse Defects4J coverage output format
        # Example: "Line coverage: 75.5%"
        line_match = re.search(r"Line coverage:\s+([\d.]+)%", output)
        if line_match:
            coverage["line_coverage"] = float(line_match.group(1))
        
        branch_match = re.search(r"Branch coverage:\s+([\d.]+)%", output)
        if branch_match:
            coverage["branch_coverage"] = float(branch_match.group(1))
        
        method_match = re.search(r"Method coverage:\s+([\d.]+)%", output)
        if method_match:
            coverage["method_coverage"] = float(method_match.group(1))
        
        return coverage


class MutationAnalyzer:
    """Analyzer for mutation testing using PIT."""
    
    def __init__(self, pit_jar: Path, project_dir: Path):
        """
        Initialize mutation analyzer.
        
        Args:
            pit_jar: Path to PIT jar file
            project_dir: Path to project directory
        """
        self.pit_jar = pit_jar
        self.project_dir = project_dir
    
    def run_mutation_testing(
        self,
        test_class: str,
        target_classes: List[str],
        timeout: int = 300
    ) -> Optional[float]:
        """
        Run mutation testing for a test class.
        
        Args:
            test_class: Test class name
            target_classes: Classes under test
            timeout: Timeout in seconds
            
        Returns:
            Mutation score (0-100) or None if failed
        """
        try:
            # Build PIT command
            cmd = [
                "java",
                "-jar", str(self.pit_jar),
                "--reportDir", str(self.project_dir / "pit-reports"),
                "--targetClasses", ",".join(target_classes),
                "--targetTests", test_class,
                "--sourceDirs", str(self.project_dir / "src" / "main" / "java"),
                "--outputFormats", "XML"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Parse mutation report
            mutation_score = self._parse_mutation_report()
            return mutation_score
        
        except subprocess.TimeoutExpired:
            logger.error(f"Mutation testing timed out for {test_class}")
            return None
        except Exception as e:
            logger.error(f"Failed to run mutation testing: {e}")
            return None
    
    def _parse_mutation_report(self) -> float:
        """
        Parse PIT XML mutation report.
        
        Returns:
            Mutation score as percentage
        """
        report_dir = self.project_dir / "pit-reports"
        xml_files = list(report_dir.glob("mutations.xml"))
        
        if not xml_files:
            logger.warning("No mutation report found")
            return 0.0
        
        try:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()
            
            # Extract mutation statistics
            mutations = root.findall(".//mutation")
            total = len(mutations)
            killed = sum(1 for m in mutations if m.get("status") == "KILLED")
            
            if total == 0:
                return 0.0
            
            return (killed / total) * 100.0
        
        except Exception as e:
            logger.error(f"Failed to parse mutation report: {e}")
            return 0.0


class TestQualityEvaluator:
    """Complete test quality evaluator combining all metrics."""
    
    def __init__(
        self,
        project_dir: Path,
        pit_jar: Optional[Path] = None
    ):
        """
        Initialize test quality evaluator.
        
        Args:
            project_dir: Path to project directory
            pit_jar: Path to PIT jar (optional)
        """
        self.project_dir = project_dir
        self.analyzer = JavaAnalyzer()
        self.coverage_analyzer = CoverageAnalyzer(project_dir)
        self.mutation_analyzer = MutationAnalyzer(pit_jar, project_dir) if pit_jar else None
    
    def evaluate_test(
        self,
        test_code: str,
        test_class_name: str,
        target_classes: Optional[List[str]] = None,
        run_mutation: bool = False
    ) -> TestMetrics:
        """
        Evaluate a test class comprehensively.
        
        Args:
            test_code: Test class source code
            test_class_name: Name of test class
            target_classes: Classes under test (for mutation testing)
            run_mutation: Whether to run mutation testing
            
        Returns:
            TestMetrics object with all metrics
        """
        metrics = TestMetrics()
        
        # 1. Check compilation
        compiles, test_file = self._compile_test(test_code, test_class_name)
        metrics.compiles = compiles
        
        if not compiles:
            logger.warning(f"Test {test_class_name} does not compile")
            return metrics
        
        # 2. Run tests
        passes = self._run_test(test_class_name)
        metrics.passes = passes
        
        # 3. Analyze code structure
        code_metrics = self.analyzer.get_code_summary(test_code)
        if code_metrics:
            metrics.cyclomatic_complexity = code_metrics.get("avg_complexity", 0.0)
            metrics.lines_of_code = code_metrics.get("total_loc", 0)
        
        # 4. Detect test smells
        metrics.test_smells = self.analyzer.detect_test_smells(test_code)
        
        # 5. Measure coverage (only if test passes)
        if passes:
            coverage = self.coverage_analyzer.run_coverage(test_class_name)
            if coverage:
                metrics.branch_coverage = coverage.get("branch_coverage", 0.0)
                metrics.line_coverage = coverage.get("line_coverage", 0.0)
                metrics.method_coverage = coverage.get("method_coverage", 0.0)
            
            # 6. Run mutation testing (optional, expensive)
            if run_mutation and self.mutation_analyzer and target_classes:
                mutation_score = self.mutation_analyzer.run_mutation_testing(
                    test_class_name,
                    target_classes
                )
                if mutation_score is not None:
                    metrics.mutation_score = mutation_score
        
        return metrics
    
    def _compile_test(self, test_code: str, test_class_name: str) -> Tuple[bool, Optional[Path]]:
        """
        Compile a test class.
        
        Args:
            test_code: Test source code
            test_class_name: Test class name
            
        Returns:
            Tuple of (success, path_to_file)
        """
        # Write test to file
        test_dir = self.project_dir / "src" / "test" / "java"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine package from code
        package_match = re.search(r"package\s+([\w.]+);", test_code)
        if package_match:
            package = package_match.group(1)
            package_path = package.replace(".", "/")
            test_file = test_dir / package_path / f"{test_class_name}.java"
        else:
            test_file = test_dir / f"{test_class_name}.java"
        
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            # Compile using Defects4J
            result = subprocess.run(
                ["defects4j", "compile"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return result.returncode == 0, test_file
        
        except Exception as e:
            logger.error(f"Failed to compile test: {e}")
            return False, None
    
    def _run_test(self, test_class_name: str) -> bool:
        """
        Run a test class.
        
        Args:
            test_class_name: Test class name
            
        Returns:
            True if all tests pass
        """
        try:
            result = subprocess.run(
                ["defects4j", "test", "-t", test_class_name],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Check if tests passed
            return "Failing tests: 0" in result.stdout
        
        except Exception as e:
            logger.error(f"Failed to run test: {e}")
            return False
    
    def compare_tests(
        self,
        original_metrics: TestMetrics,
        refined_metrics: TestMetrics
    ) -> Dict[str, float]:
        """
        Compare two test versions.
        
        Args:
            original_metrics: Metrics from original test
            refined_metrics: Metrics from refined test
            
        Returns:
            Dict with improvement deltas
        """
        return {
            # Effectiveness improvements (positive = better)
            "coverage_delta": refined_metrics.branch_coverage - original_metrics.branch_coverage,
            "mutation_delta": refined_metrics.mutation_score - original_metrics.mutation_score,
            
            # Maintainability improvements (negative = better for complexity/smells)
            "complexity_delta": original_metrics.cyclomatic_complexity - refined_metrics.cyclomatic_complexity,
            "smells_delta": sum(original_metrics.test_smells.values()) - sum(refined_metrics.test_smells.values()),
            
            # Compilation status
            "still_compiles": refined_metrics.compiles,
            "still_passes": refined_metrics.passes
        }
