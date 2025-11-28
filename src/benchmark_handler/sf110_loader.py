import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProjectInfo:
    """Information about a project in the dataset."""
    name: str
    project_dir: Path
    jar_files: List[Path]
    # Source files might not be available in binary datasets
    source_files: List[Path] = None

class JARBenchmarkLoader:
    """
    Loader for JAR-based benchmarks (SF110, Extended-DynaMOSA).
    """
    
    def __init__(self, dataset_home: str):
        """
        Initialize the loader.
        
        Args:
            dataset_home: Path to the dataset root directory
        """
        self.dataset_home = Path(dataset_home)
        if not self.dataset_home.exists():
            raise ValueError(f"Dataset home not found: {dataset_home}")
            
    def get_projects(self) -> List[str]:
        """
        Get list of project names in the dataset.
        
        Returns:
            List of project directory names
        """
        return [
            d.name for d in self.dataset_home.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
        
    def load_project(self, project_name: str) -> Optional[ProjectInfo]:
        """
        Load a specific project.
        
        Args:
            project_name: Name of the project directory
            
        Returns:
            ProjectInfo object or None if not found
        """
        project_dir = self.dataset_home / project_name
        if not project_dir.exists():
            logger.error(f"Project {project_name} not found in {self.dataset_home}")
            return None
            
        # Find JAR files
        jar_files = list(project_dir.glob("*.jar"))
        
        # Also look in lib/ if it exists
        lib_dir = project_dir / "lib"
        if lib_dir.exists():
            jar_files.extend(lib_dir.glob("*.jar"))
            
        return ProjectInfo(
            name=project_name,
            project_dir=project_dir,
            jar_files=jar_files,
            source_files=[] # Placeholder as we are in binary mode
        )

    def get_classpath(self, project_info: ProjectInfo) -> str:
        """
        Get the classpath for the project.
        
        Args:
            project_info: Project info object
            
        Returns:
            Classpath string
        """
        return os.pathsep.join([str(jar) for jar in project_info.jar_files])
