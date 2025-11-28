import logging
from pathlib import Path
from typing import Optional, List
from .sf110_loader import JARBenchmarkLoader, ProjectInfo

logger = logging.getLogger(__name__)

class BenchmarkLoader:
    """
    Unified loader for project benchmarks.
    Currently supports SF110 and Extended-DynaMOSA (JAR-based).
    """
    
    def __init__(self, sf110_home: str, extended_dynamosa_home: str):
        self.sf110_loader = JARBenchmarkLoader(sf110_home)
        self.dynamosa_loader = JARBenchmarkLoader(extended_dynamosa_home)
        
    def load_project(self, project_name: str, dataset: str = "sf110") -> Optional[ProjectInfo]:
        """
        Load a project from the specified dataset.
        
        Args:
            project_name: Name of the project
            dataset: 'sf110' or 'dynamosa'
            
        Returns:
            ProjectInfo or None
        """
        if dataset.lower() == "sf110":
            return self.sf110_loader.load_project(project_name)
        elif dataset.lower() == "dynamosa":
            return self.dynamosa_loader.load_project(project_name)
        else:
            logger.error(f"Unknown dataset: {dataset}")
            return None

    def get_projects(self, dataset: str = "sf110") -> List[str]:
        if dataset.lower() == "sf110":
            return self.sf110_loader.get_projects()
        elif dataset.lower() == "dynamosa":
            return self.dynamosa_loader.get_projects()
        return []