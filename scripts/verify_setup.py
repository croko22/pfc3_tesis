import sys
print("Script started")
import logging
print("Logging imported")
from pathlib import Path
print("Pathlib imported")

# Add src to path
sys.path.append(str(Path.cwd() / "src"))
print("Path updated")

from benchmark_handler.loader import BenchmarkLoader
print("Loader imported")
from evaluation.metrics import TestQualityEvaluator
print("Evaluator imported")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify():
    # 1. Test Loader
    logger.info("Testing BenchmarkLoader...")
    loader = BenchmarkLoader(
        sf110_home="data/SF110-binary",
        extended_dynamosa_home="data/extended-dynamosa-repos-binary"
    )
    
    projects = loader.get_projects("sf110")
    logger.info(f"Found {len(projects)} SF110 projects")
    
    if not projects:
        logger.error("No projects found!")
        return
        
    project_name = projects[0]
    logger.info(f"Loading project: {project_name}")
    project = loader.load_project(project_name, "sf110")
    
    if not project:
        logger.error("Failed to load project")
        return
        
    logger.info(f"Project loaded: {project.name}")
    logger.info(f"JARs found: {len(project.jar_files)}")
    
    # 2. Test Metrics (Compilation)
    logger.info("Testing TestQualityEvaluator...")
    
    # Create a dummy test
    dummy_test = f"""
    import org.junit.Test;
    import static org.junit.Assert.*;
    
    public class DummyTest {{
        @Test
        public void testSomething() {{
            assertTrue(true);
        }}
    }}
    """
    
    classpath = loader.sf110_loader.get_classpath(project)
    evaluator = TestQualityEvaluator(Path.cwd(), classpath)
    
    metrics = evaluator.evaluate_test(dummy_test, "DummyTest")
    
    logger.info(f"Compilation success: {metrics.compiles}")
    logger.info(f"Execution success: {metrics.passes}")
    
    if metrics.compiles and metrics.passes:
        logger.info("VERIFICATION SUCCESSFUL")
    else:
        logger.error("VERIFICATION FAILED")

if __name__ == "__main__":
    verify()
