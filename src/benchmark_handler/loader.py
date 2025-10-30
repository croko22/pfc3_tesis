import os
import subprocess
import logging
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DefectInfo:
    """Information about a specific defect from Defects4J."""
    project: str
    bug_id: int
    buggy_classes: List[str]
    fixed_classes: List[str]
    modified_classes: List[str]


class Defects4JLoader:
    """
    Loader for the Defects4J benchmark.
    
    Provides functionality to checkout projects, extract source files,
    and manage buggy/fixed versions.
    """
    
    def __init__(self, defects4j_home: str, workspace: str):
        """
        Initialize the Defects4J loader.
        
        Args:
            defects4j_home: Path to Defects4J installation
            workspace: Working directory for checkouts
        """
        self.defects4j_home = Path(defects4j_home)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Verify Defects4J is installed
        if not self.defects4j_home.exists():
            raise ValueError(f"Defects4J home not found: {defects4j_home}")
        
        self.defects4j_cmd = self.defects4j_home / "framework" / "bin" / "defects4j"
        if not self.defects4j_cmd.exists():
            raise ValueError(f"Defects4J command not found: {self.defects4j_cmd}")
    
    def get_project_bugs(self, project: str) -> List[int]:
        """
        Get list of bug IDs for a project.
        
        Args:
            project: Project name (e.g., 'Chart', 'Lang')
            
        Returns:
            List of bug IDs
        """
        try:
            result = subprocess.run(
                [str(self.defects4j_cmd), "bids", "-p", project],
                capture_output=True,
                text=True,
                check=True
            )
            bug_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            return bug_ids
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get bugs for {project}: {e}")
            return []
    
    def checkout_bug(self, project: str, bug_id: int, version: str = "b") -> Optional[Path]:
        """
        Checkout a specific bug version.
        
        Args:
            project: Project name
            bug_id: Bug ID
            version: 'b' for buggy, 'f' for fixed
            
        Returns:
            Path to checked out project, or None if failed
        """
        checkout_dir = self.workspace / f"{project}_{bug_id}{version}"
        
        # Remove existing checkout
        if checkout_dir.exists():
            import shutil
            shutil.rmtree(checkout_dir)
        
        try:
            subprocess.run(
                [
                    str(self.defects4j_cmd),
                    "checkout",
                    "-p", project,
                    "-v", f"{bug_id}{version}",
                    "-w", str(checkout_dir)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Checked out {project}-{bug_id}{version} to {checkout_dir}")
            return checkout_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout {project}-{bug_id}{version}: {e}")
            return None
    
    def get_trigger_tests(self, project_dir: Path) -> List[str]:
        """
        Get list of trigger tests for a bug.
        
        Args:
            project_dir: Path to checked out project
            
        Returns:
            List of test class names that trigger the bug
        """
        try:
            result = subprocess.run(
                [str(self.defects4j_cmd), "export", "-p", "tests.trigger"],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                check=True
            )
            tests = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return tests
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get trigger tests: {e}")
            return []
    
    def get_modified_classes(self, project_dir: Path) -> List[str]:
        """
        Get list of modified classes for a bug.
        
        Args:
            project_dir: Path to checked out project
            
        Returns:
            List of modified class names
        """
        try:
            result = subprocess.run(
                [str(self.defects4j_cmd), "export", "-p", "classes.modified"],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                check=True
            )
            classes = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return classes
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get modified classes: {e}")
            return []
    
    def compile_project(self, project_dir: Path) -> bool:
        """
        Compile the project.
        
        Args:
            project_dir: Path to checked out project
            
        Returns:
            True if compilation succeeded
        """
        try:
            subprocess.run(
                [str(self.defects4j_cmd), "compile"],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Compiled project at {project_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile project: {e}")
            return False
    
    def run_tests(self, project_dir: Path, test_class: Optional[str] = None) -> Dict:
        """
        Run tests for the project.
        
        Args:
            project_dir: Path to checked out project
            test_class: Specific test class to run (None = all tests)
            
        Returns:
            Dict with test results
        """
        cmd = [str(self.defects4j_cmd), "test"]
        if test_class:
            cmd.extend(["-t", test_class])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse test results
            output = result.stdout
            return {
                "passed": "Failing tests: 0" in output,
                "output": output,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Test execution timed out")
            return {"passed": False, "output": "Timeout", "returncode": -1}
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run tests: {e}")
            return {"passed": False, "output": str(e), "returncode": e.returncode}
    
    def get_source_files(self, project_dir: Path) -> Iterator[Tuple[str, str]]:
        """
        Get all source files from a checked out project.
        
        Args:
            project_dir: Path to checked out project
            
        Yields:
            Tuples of (file_path, file_content)
        """
        # Common source directories in Defects4J projects
        src_dirs = [
            project_dir / "src" / "java",
            project_dir / "src" / "main" / "java",
            project_dir / "source" / "java",
        ]
        
        for src_dir in src_dirs:
            if not src_dir.exists():
                continue
            
            for root, dirs, files in os.walk(src_dir):
                # Skip test directories
                if 'test' in root.lower():
                    continue
                
                for file in files:
                    if file.endswith('.java'):
                        file_path = Path(root) / file
                        
                        # Skip test files by name
                        if file.endswith('Test.java') or file.endswith('Tests.java'):
                            continue
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            yield (str(file_path), content)
                        except Exception as e:
                            logger.error(f"Error reading {file_path}: {e}")
    
    def get_class_path(self, project_dir: Path, class_name: str) -> Optional[Path]:
        """
        Find the source file for a given class name.
        
        Args:
            project_dir: Path to checked out project
            class_name: Fully qualified class name (e.g., 'org.jfree.chart.ChartPanel')
            
        Returns:
            Path to source file, or None if not found
        """
        # Convert class name to file path
        relative_path = class_name.replace('.', os.sep) + '.java'
        
        src_dirs = [
            project_dir / "src" / "java",
            project_dir / "src" / "main" / "java",
            project_dir / "source" / "java",
        ]
        
        for src_dir in src_dirs:
            file_path = src_dir / relative_path
            if file_path.exists():
                return file_path
        
        return None
    
    def load_benchmark(
        self,
        projects: List[str],
        bug_ids: Optional[List[int]] = None,
        max_bugs_per_project: int = 0
    ) -> Iterator[Dict]:
        """
        Load Defects4J benchmark projects.
        
        Args:
            projects: List of project names to load
            bug_ids: Specific bug IDs to load (None = all)
            max_bugs_per_project: Maximum bugs per project (0 = unlimited)
            
        Yields:
            Dict containing bug information and paths
        """
        for project in projects:
            logger.info(f"Loading project: {project}")
            
            # Get all bugs for project
            all_bugs = self.get_project_bugs(project)
            
            if bug_ids:
                bugs_to_load = [b for b in all_bugs if b in bug_ids]
            else:
                bugs_to_load = all_bugs
            
            if max_bugs_per_project > 0:
                bugs_to_load = bugs_to_load[:max_bugs_per_project]
            
            for bug_id in bugs_to_load:
                logger.info(f"Loading {project}-{bug_id}")
                
                # Checkout buggy version
                buggy_dir = self.checkout_bug(project, bug_id, "b")
                if not buggy_dir:
                    logger.warning(f"Skipping {project}-{bug_id}: checkout failed")
                    continue
                
                # Compile
                if not self.compile_project(buggy_dir):
                    logger.warning(f"Skipping {project}-{bug_id}: compilation failed")
                    continue
                
                # Get metadata
                modified_classes = self.get_modified_classes(buggy_dir)
                trigger_tests = self.get_trigger_tests(buggy_dir)
                
                yield {
                    "project": project,
                    "bug_id": bug_id,
                    "buggy_dir": buggy_dir,
                    "modified_classes": modified_classes,
                    "trigger_tests": trigger_tests,
                    "source_files": list(self.get_source_files(buggy_dir))
                }