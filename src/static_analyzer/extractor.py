"""
Static analyzer for Java code.

Extracts AST information, computes complexity metrics, and analyzes code structure.
Uses javalang for Java parsing.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import javalang
from javalang.tree import *

logger = logging.getLogger(__name__)


@dataclass
class MethodInfo:
    """Information about a Java method."""
    name: str
    parameters: List[str]
    return_type: str
    modifiers: Set[str]
    body: str
    start_position: Optional[int]
    cyclomatic_complexity: int
    lines_of_code: int


@dataclass
class ClassInfo:
    """Information about a Java class."""
    name: str
    package: str
    modifiers: Set[str]
    methods: List[MethodInfo]
    fields: List[str]
    imports: List[str]
    extends: Optional[str]
    implements: List[str]


class JavaAnalyzer:
    """Analyzer for Java source code using AST parsing."""
    
    def __init__(self):
        """Initialize the Java analyzer."""
        pass
    
    def parse(self, source_code: str) -> Optional[javalang.tree.CompilationUnit]:
        """
        Parse Java source code into an AST.
        
        Args:
            source_code: Java source code as string
            
        Returns:
            Parsed AST or None if parsing fails
        """
        try:
            tree = javalang.parse.parse(source_code)
            return tree
        except Exception as e:
            logger.error(f"Failed to parse Java code: {e}")
            return None
    
    def extract_class_info(self, source_code: str) -> Optional[ClassInfo]:
        """
        Extract comprehensive information about a Java class.
        
        Args:
            source_code: Java source code
            
        Returns:
            ClassInfo object or None if extraction fails
        """
        tree = self.parse(source_code)
        if not tree:
            return None
        
        try:
            # Extract package
            package = tree.package.name if tree.package else ""
            
            # Extract imports
            imports = [imp.path for imp in tree.imports] if tree.imports else []
            
            # Find the main class declaration
            class_decl = None
            for path, node in tree.filter(ClassDeclaration):
                class_decl = node
                break
            
            if not class_decl:
                logger.warning("No class declaration found")
                return None
            
            # Extract class information
            class_name = class_decl.name
            modifiers = set(class_decl.modifiers) if class_decl.modifiers else set()
            extends = class_decl.extends.name if class_decl.extends else None
            implements = [impl.name for impl in class_decl.implements] if class_decl.implements else []
            
            # Extract methods
            methods = []
            for method_decl in class_decl.methods:
                method_info = self.extract_method_info(method_decl, source_code)
                if method_info:
                    methods.append(method_info)
            
            # Extract fields
            fields = []
            for field_decl in class_decl.fields:
                for declarator in field_decl.declarators:
                    fields.append(declarator.name)
            
            return ClassInfo(
                name=class_name,
                package=package,
                modifiers=modifiers,
                methods=methods,
                fields=fields,
                imports=imports,
                extends=extends,
                implements=implements
            )
        
        except Exception as e:
            logger.error(f"Failed to extract class info: {e}")
            return None
    
    def extract_method_info(self, method_decl: MethodDeclaration, source_code: str) -> Optional[MethodInfo]:
        """
        Extract information about a specific method.
        
        Args:
            method_decl: Method declaration node from AST
            source_code: Full source code (for extracting method body)
            
        Returns:
            MethodInfo object or None if extraction fails
        """
        try:
            name = method_decl.name
            modifiers = set(method_decl.modifiers) if method_decl.modifiers else set()
            return_type = method_decl.return_type.name if method_decl.return_type else "void"
            
            # Extract parameters
            parameters = []
            if method_decl.parameters:
                for param in method_decl.parameters:
                    param_type = param.type.name if hasattr(param.type, 'name') else str(param.type)
                    parameters.append(f"{param_type} {param.name}")
            
            # Calculate cyclomatic complexity
            complexity = self.calculate_cyclomatic_complexity(method_decl)
            
            # Estimate lines of code (rough approximation)
            loc = self.estimate_method_loc(method_decl)
            
            return MethodInfo(
                name=name,
                parameters=parameters,
                return_type=return_type,
                modifiers=modifiers,
                body="",  # We don't extract body text for efficiency
                start_position=method_decl.position.line if method_decl.position else None,
                cyclomatic_complexity=complexity,
                lines_of_code=loc
            )
        
        except Exception as e:
            logger.error(f"Failed to extract method info: {e}")
            return None
    
    def calculate_cyclomatic_complexity(self, method_decl: MethodDeclaration) -> int:
        """
        Calculate cyclomatic complexity for a method.
        
        McCabe's formula: M = E - N + 2P
        For a single method: M = decision_points + 1
        
        Args:
            method_decl: Method declaration node
            
        Returns:
            Cyclomatic complexity value
        """
        if not method_decl.body:
            return 1
        
        decision_points = 0
        
        # Count decision points
        for path, node in method_decl.body:
            if isinstance(node, (IfStatement, ForStatement, WhileStatement, DoStatement)):
                decision_points += 1
            elif isinstance(node, SwitchStatement):
                # Each case is a decision point
                decision_points += len(node.cases) if node.cases else 1
            elif isinstance(node, TernaryExpression):
                decision_points += 1
            elif isinstance(node, BinaryOperation):
                # Count logical operators (&&, ||)
                if node.operator in ['&&', '||']:
                    decision_points += 1
        
        return decision_points + 1
    
    def estimate_method_loc(self, method_decl: MethodDeclaration) -> int:
        """
        Estimate lines of code for a method.
        
        Args:
            method_decl: Method declaration node
            
        Returns:
            Estimated lines of code
        """
        if not method_decl.body:
            return 1
        
        # Count statements as a proxy for LOC
        statement_count = 0
        for path, node in method_decl.body:
            if isinstance(node, Statement):
                statement_count += 1
        
        return max(statement_count, 1)
    
    def extract_test_methods(self, source_code: str) -> List[MethodInfo]:
        """
        Extract test methods from a test class.
        
        Args:
            source_code: Test class source code
            
        Returns:
            List of test method information
        """
        tree = self.parse(source_code)
        if not tree:
            return []
        
        test_methods = []
        
        for path, node in tree.filter(MethodDeclaration):
            # Check if method has @Test annotation
            is_test = False
            if node.annotations:
                for annotation in node.annotations:
                    if annotation.name == "Test":
                        is_test = True
                        break
            
            # Also check method name patterns
            if node.name.startswith("test") or is_test:
                method_info = self.extract_method_info(node, source_code)
                if method_info:
                    test_methods.append(method_info)
        
        return test_methods
    
    def detect_test_smells(self, source_code: str) -> Dict[str, int]:
        """
        Detect test smells in test code.
        
        Args:
            source_code: Test class source code
            
        Returns:
            Dict mapping smell name to count
        """
        tree = self.parse(source_code)
        if not tree:
            return {}
        
        smells = {
            "Assertion Roulette": 0,
            "Eager Test": 0,
            "Mystery Guest": 0,
            "Resource Optimism": 0,
            "Verbose Test": 0,
            "Sleepy Test": 0,
            "Duplicate Assert": 0,
            "Unknown Test": 0,
            "Ignored Test": 0,
            "Empty Test": 0
        }
        
        test_methods = self.extract_test_methods(source_code)
        
        for method_info in test_methods:
            # Empty Test: no assertions
            method_node = self._find_method_node(tree, method_info.name)
            if not method_node or not method_node.body:
                smells["Empty Test"] += 1
                continue
            
            # Count assertions
            assertions = self._count_assertions(method_node)
            
            # Assertion Roulette: multiple assertions without explanation
            if assertions > 3:
                smells["Assertion Roulette"] += 1
            
            # Empty Test
            if assertions == 0 and method_info.lines_of_code > 1:
                smells["Unknown Test"] += 1
            
            # Verbose Test: high complexity
            if method_info.cyclomatic_complexity > 10:
                smells["Verbose Test"] += 1
            
            # Eager Test: testing multiple methods
            if assertions > 5:
                smells["Eager Test"] += 1
            
            # Sleepy Test: Thread.sleep() calls
            if self._contains_sleep(method_node):
                smells["Sleepy Test"] += 1
            
            # Mystery Guest: file/database access
            if self._contains_external_resource(method_node):
                smells["Mystery Guest"] += 1
            
            # Ignored Test: @Ignore annotation
            if method_node.annotations:
                for annotation in method_node.annotations:
                    if annotation.name in ["Ignore", "Disabled"]:
                        smells["Ignored Test"] += 1
        
        return smells
    
    def _find_method_node(self, tree: CompilationUnit, method_name: str) -> Optional[MethodDeclaration]:
        """Find a method node by name."""
        for path, node in tree.filter(MethodDeclaration):
            if node.name == method_name:
                return node
        return None
    
    def _count_assertions(self, method_node: MethodDeclaration) -> int:
        """Count assertion calls in a method."""
        count = 0
        if not method_node.body:
            return 0
        
        for path, node in method_node.body:
            if isinstance(node, MethodInvocation):
                if node.member and node.member.startswith("assert"):
                    count += 1
        
        return count
    
    def _contains_sleep(self, method_node: MethodDeclaration) -> bool:
        """Check if method contains Thread.sleep() calls."""
        if not method_node.body:
            return False
        
        for path, node in method_node.body:
            if isinstance(node, MethodInvocation):
                if node.member == "sleep" or "sleep" in str(node):
                    return True
        
        return False
    
    def _contains_external_resource(self, method_node: MethodDeclaration) -> bool:
        """Check if method accesses external resources."""
        if not method_node.body:
            return False
        
        resource_indicators = ["File", "FileReader", "FileWriter", "Connection", "InputStream", "OutputStream"]
        
        for path, node in method_node.body:
            node_str = str(node)
            for indicator in resource_indicators:
                if indicator in node_str:
                    return True
        
        return False
    
    def get_code_summary(self, source_code: str) -> Dict:
        """
        Get a summary of code metrics.
        
        Args:
            source_code: Java source code
            
        Returns:
            Dict with summary metrics
        """
        class_info = self.extract_class_info(source_code)
        if not class_info:
            return {}
        
        total_complexity = sum(m.cyclomatic_complexity for m in class_info.methods)
        total_loc = sum(m.lines_of_code for m in class_info.methods)
        
        return {
            "class_name": class_info.name,
            "package": class_info.package,
            "num_methods": len(class_info.methods),
            "num_fields": len(class_info.fields),
            "total_complexity": total_complexity,
            "avg_complexity": total_complexity / len(class_info.methods) if class_info.methods else 0,
            "total_loc": total_loc,
            "avg_loc": total_loc / len(class_info.methods) if class_info.methods else 0
        }
