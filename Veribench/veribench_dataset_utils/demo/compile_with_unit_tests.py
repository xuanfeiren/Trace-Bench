"""
Demo: Compile Lean code with unit tests using eval_utils

This demo shows how to:
1. Extract implementation from generated Lean code
2. Combine it with unit tests from gold reference (task_2.json)
3. Compile the combined code to verify unit test compatibility

This mirrors the VeribenchGuidewithUnitTests workflow from guide.py
"""

import sys
from pathlib import Path

# Add parent directory (veribench_dataset_utils) and project root to path
parent_dir = Path(__file__).resolve().parent.parent
project_root = parent_dir.parent

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_root))

from eval_utils import compile, combine_code_with_tests

# Example Lean code for task 2 (CountingSort implementation only, no tests)
# This simulates what an LLM might generate - just the implementation
lean_code_implementation = """/-!
# CountingSort Implementation

This module implements counting sort algorithm for lists of natural numbers.
-/

namespace CountingSort

/--
Counting sort implementation that sorts a list of natural numbers.
It counts occurrences of each element and uses this to place elements in sorted order.
-/
def countingSort (arr : List Nat) : List Nat := Id.run do
  match arr with
  | [] => []
  | _ => 
    let maxVal := arr.foldl max 0
    -- Create count array initialized to 0s
    let mut count := mkArray (maxVal + 1) 0
    -- Count occurrences
    for x in arr do
      count := count.modify x (· + 1)
    -- Build result by expanding counts
    let mut result := []
    for i in [:maxVal + 1] do
      let cnt := count[i]!  -- Use ! operator for array access
      result := result ++ List.replicate cnt i
    return result

/-- Returns true if list is sorted in ascending order -/
def isSorted : List Nat → Bool
| [] => true  
| [_] => true
| x :: y :: xs => x ≤ y && isSorted (y :: xs)

/-- Returns number of occurrences of element in list -/
def countOccurrences (xs : List Nat) (x : Nat) : Nat :=
  xs.filter (· = x) |>.length

/-- Returns true if two lists are permutations of each other -/
def isPerm (xs ys : List Nat) : Bool :=
  xs.length = ys.length && 
  (xs.foldl (fun acc x => acc && countOccurrences ys x = countOccurrences xs x) true)

/-- Pre-condition: list contains natural numbers (always true by type system) -/
def Pre (arr : List Nat) : Prop := True 

/-- Post-condition: output list is sorted and is a permutation of input -/
def Post (arr result : List Nat) : Prop :=
  isSorted result ∧ isPerm arr result 

/-- Correctness theorem -/
theorem correctness (arr : List Nat) (h : Pre arr) :
  Post arr (countingSort arr) := sorry

end CountingSort
"""

def main():
    task_id = 2  # CountingSort is task 2
    
    print("=" * 80)
    print(f"Demo: Compiling Lean Code with Unit Tests (Task {task_id})")
    print("=" * 80)
    print()
    
    # Step 1: Compile the implementation alone
    print("Step 1: Compiling implementation only...")
    print("-" * 80)
    compile_score, compile_feedback = compile(lean_code_implementation)
    print(f"Compilation Score: {compile_score}")
    print(f"Feedback: {compile_feedback[:200]}...")
    print()
    
    if compile_score == 0.0:
        print("✗ FAILED: Implementation has compilation errors.")
        print("Cannot proceed to unit tests with broken implementation.")
        return
    
    print("✓ Implementation compiles successfully!")
    print()
    
    # Step 2: Combine with unit tests from gold reference
    print("Step 2: Combining implementation with unit tests from gold reference...")
    print("-" * 80)
    try:
        combined_code = combine_code_with_tests(task_id, lean_code_implementation)
        print("✓ Successfully combined implementation with unit tests")
        print()
        print("Combined code preview:")
        print(combined_code)
        print()
    except Exception as e:
        print(f"✗ FAILED to combine code: {e}")
        return
    
    # Step 3: Compile the combined code with unit tests
    print("Step 3: Compiling combined code with unit tests...")
    print("-" * 80)
    unit_test_score, unit_test_feedback = compile(combined_code)
    print(f"Unit Test Score: {unit_test_score}")
    print()
    print("Unit Test Feedback:")
    print(unit_test_feedback[:500] + ("..." if len(unit_test_feedback) > 500 else ""))
    print()
    
    # Step 4: Report final result
    print("=" * 80)
    print("Final Result:")
    print("=" * 80)
    if unit_test_score == 1.0:
        print("✓ SUCCESS: All unit tests passed!")
        print("The implementation correctly satisfies the gold reference unit tests.")
    elif compile_score == 1.0 and unit_test_score == 0.0:
        print("⚠ PARTIAL: Implementation compiles but fails unit tests.")
        print("Score: 0.5 (compiles but doesn't match expected behavior)")
    else:
        print("✗ FAILED: Both compilation and unit tests failed.")
    print()
    
    print(f"Final Score: {unit_test_score if compile_score == 1.0 else 0.0}")

if __name__ == "__main__":
    main()

