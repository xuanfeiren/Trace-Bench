"""
Test script to demonstrate getting unit tests by task_id.
"""

from unit_tests import get_unit_tests
from create_datasets import load_task_full


def demo_get_unit_tests():
    """Demonstrate getting unit tests for different tasks."""
    
    print("=" * 80)
    print("Demo: Get Unit Tests by Task ID")
    print("=" * 80)
    print()
    
    # Test task 0 (binary_search)
    print("Task 0: Binary Search")
    print("-" * 80)
    tests = get_unit_tests(0)
    print(f"Number of unit tests: {len(tests)}")
    if tests:
        print(f"\nFirst 3 tests:")
        for i, test in enumerate(tests[:3]):
            print(f"  {i+1}. {test}")
    print()
    
    # Test task 3 (edit_distance)
    print("Task 3: Edit Distance")
    print("-" * 80)
    tests = get_unit_tests(3)
    print(f"Number of unit tests: {len(tests)}")
    if tests:
        print(f"\nFirst 3 tests:")
        for i, test in enumerate(tests[:3]):
            print(f"  {i+1}. {test}")
    print()
    
    # Test multiple tasks
    print("Unit Test Statistics for First 10 Tasks:")
    print("-" * 80)
    for task_id in range(10):
        tests = get_unit_tests(task_id)
        task_data = load_task_full(task_id)
        print(f"Task {task_id:3d}: {len(tests):2d} tests")
    print()
    
    # Show a complete workflow example
    print("=" * 80)
    print("Complete Workflow Example:")
    print("=" * 80)
    print()
    
    task_id = 0
    
    # Step 1: Load task data
    task = load_task_full(task_id)
    print(f"Step 1: Load task {task_id}")
    print(f"  Python code: {len(task['python_code'])} chars")
    print(f"  Gold Lean4 code: {len(task['gold_reference_lean4_code'])} chars")
    print()
    
    # Step 2: Get unit tests
    tests = get_unit_tests(task_id)
    print(f"Step 2: Get unit tests")
    print(f"  Found {len(tests)} unit tests")
    print()
    
    # Step 3: Show what the tests look like
    print(f"Step 3: Examine unit tests")
    for i, test in enumerate(tests[:5]):
        print(f"  Test {i+1}: {test[:80]}{'...' if len(test) > 80 else ''}")
    print()
    
    print("=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    demo_get_unit_tests()

