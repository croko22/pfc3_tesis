"""
Example usage of GSPO-UTG components.

This script demonstrates how to use individual components of the system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.static_analyzer.extractor import JavaAnalyzer
from src.evaluation.metrics import TestMetrics
from src.rl_env.reward import RewardFunction, RewardConfig


def example_java_analysis():
    """Example: Analyze Java code."""
    print("=" * 80)
    print("EXAMPLE: Java Code Analysis")
    print("=" * 80)
    
    # Sample Java code
    java_code = """
    package com.example;
    
    public class Calculator {
        public int add(int a, int b) {
            return a + b;
        }
        
        public int multiply(int a, int b) {
            int result = 0;
            for (int i = 0; i < b; i++) {
                result += a;
            }
            return result;
        }
    }
    """
    
    # Analyze
    analyzer = JavaAnalyzer()
    class_info = analyzer.extract_class_info(java_code)
    
    if class_info:
        print(f"\nClass: {class_info.name}")
        print(f"Package: {class_info.package}")
        print(f"Methods: {len(class_info.methods)}")
        
        for method in class_info.methods:
            print(f"\n  Method: {method.name}")
            print(f"    Parameters: {method.parameters}")
            print(f"    Complexity: {method.cyclomatic_complexity}")
            print(f"    LOC: {method.lines_of_code}")
    
    print("\n")


def example_test_smell_detection():
    """Example: Detect test smells."""
    print("=" * 80)
    print("EXAMPLE: Test Smell Detection")
    print("=" * 80)
    
    # Sample test with smells
    test_code = """
    package com.example;
    
    import org.junit.Test;
    import static org.junit.Assert.*;
    
    public class CalculatorTest {
        @Test
        public void testMultipleMethods() {
            Calculator calc = new Calculator();
            
            // Testing add
            assertEquals(5, calc.add(2, 3));
            assertEquals(10, calc.add(5, 5));
            
            // Testing multiply (Eager Test smell)
            assertEquals(6, calc.multiply(2, 3));
            assertEquals(10, calc.multiply(2, 5));
            
            // Multiple assertions without messages (Assertion Roulette)
            assertTrue(calc.add(1, 1) > 0);
            assertTrue(calc.multiply(2, 2) > 0);
            assertFalse(calc.add(-1, -1) > 0);
        }
        
        @Test
        public void testWithSleep() throws Exception {
            Calculator calc = new Calculator();
            Thread.sleep(1000);  // Sleepy Test smell
            assertEquals(5, calc.add(2, 3));
        }
    }
    """
    
    # Detect smells
    analyzer = JavaAnalyzer()
    smells = analyzer.detect_test_smells(test_code)
    
    print("\nDetected Test Smells:")
    for smell, count in smells.items():
        if count > 0:
            print(f"  {smell}: {count}")
    
    print("\n")


def example_reward_calculation():
    """Example: Calculate reward for test refinement."""
    print("=" * 80)
    print("EXAMPLE: Reward Calculation")
    print("=" * 80)
    
    # Create sample metrics
    original_metrics = TestMetrics(
        branch_coverage=75.0,
        mutation_score=60.0,
        test_smells={"Assertion Roulette": 2, "Eager Test": 1},
        cyclomatic_complexity=8.0,
        lines_of_code=50,
        compiles=True,
        passes=True
    )
    
    refined_metrics = TestMetrics(
        branch_coverage=76.0,
        mutation_score=61.0,
        test_smells={"Assertion Roulette": 0, "Eager Test": 0},
        cyclomatic_complexity=4.0,
        lines_of_code=45,
        compiles=True,
        passes=True
    )
    
    # Calculate reward
    config = RewardConfig()
    reward_fn = RewardFunction(config)
    
    reward = reward_fn.calculate_reward(original_metrics, refined_metrics)
    feedback = reward_fn.get_detailed_feedback(original_metrics, refined_metrics)
    
    print(f"\nTotal Reward: {reward:.4f}")
    print("\nReward Breakdown:")
    print(f"  Smell Reduction: {feedback['smells_reward']:.4f}")
    print(f"  Complexity Reduction: {feedback['complexity_reward']:.4f}")
    print(f"  Coverage Improvement: {feedback['coverage_reward']:.4f}")
    print(f"  Mutation Improvement: {feedback['mutation_reward']:.4f}")
    
    print("\nMetric Deltas:")
    print(f"  Smells: {feedback['smell_delta']:+.0f}")
    print(f"  Complexity: {feedback['complexity_delta']:+.1f}")
    print(f"  Coverage: {feedback['coverage_delta']:+.1f}%")
    print(f"  Mutation: {feedback['mutation_delta']:+.1f}%")
    
    print("\n")


def example_prompt_generation():
    """Example: Generate prompt for LLM."""
    print("=" * 80)
    print("EXAMPLE: LLM Prompt Generation")
    print("=" * 80)
    
    source_code = """
    public class StringUtils {
        public static boolean isEmpty(String str) {
            return str == null || str.length() == 0;
        }
    }
    """
    
    original_test = """
    @Test
    public void test1() {
        assertTrue(StringUtils.isEmpty(null));
        assertTrue(StringUtils.isEmpty(""));
        assertFalse(StringUtils.isEmpty("a"));
    }
    """
    
    from src.rl_env.environment import TestRefinementState
    
    state = TestRefinementState(
        original_test_code=original_test,
        source_code=source_code,
        source_class_name="StringUtils"
    )
    
    prompt = state.to_prompt()
    
    print("\nGenerated Prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    print("\n")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "GSPO-UTG EXAMPLES" + " " * 41 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    try:
        example_java_analysis()
        example_test_smell_detection()
        example_reward_calculation()
        example_prompt_generation()
        
        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
