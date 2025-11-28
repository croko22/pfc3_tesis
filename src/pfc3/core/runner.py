import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
from .config import cfg

class CommandRunner:
    """Standardized execution of Java commands."""
    
    def run_java(self, 
                 classpath: str, 
                 main_class: str, 
                 args: List[str] = None, 
                 timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a Java class."""
        cmd = ["java", "-cp", classpath, main_class]
        if args:
            cmd.extend(args)
            
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
    def run_javac(self,
                  classpath: str,
                  source_files: List[Path],
                  output_dir: Path,
                  timeout: int = 60) -> subprocess.CompletedProcess:
        """Compile Java files."""
        cmd = [
            "javac",
            "-cp", classpath,
            "-d", str(output_dir)
        ] + [str(f) for f in source_files]
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
    def run_evosuite(self,
                     target_jar: Path,
                     class_name: str,
                     output_dir: Path,
                     time_budget: int = 60) -> Dict:
        """Run EvoSuite generation."""
        cmd = [
            "java", "-jar", str(cfg.evosuite_jar),
            "-class", class_name,
            "-target", str(target_jar),
            "-Dtest_dir", str(output_dir),
            "-Dsearch_budget", str(time_budget),
            "-Dminimize", "true",
            "-Dassertion_strategy", "all"
        ]
        
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=time_budget + 120
            )
            elapsed = time.time() - start
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "time": elapsed
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "timeout", "time": time.time() - start}

runner = CommandRunner()
