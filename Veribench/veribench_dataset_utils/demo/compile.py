"""
Demo: Compile Lean code using eval_utils.compile()

This demo shows how to use the compile() function from eval_utils.py
to compile Lean code and get compilation feedback.
"""

import sys
from pathlib import Path

# Add parent directory (veribench_dataset_utils) and project root to path
parent_dir = Path(__file__).resolve().parent.parent
project_root = parent_dir.parent

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_root))

from eval_utils import compile

# Example Lean code for task 2 (CountingSort)
lean_code = """/-!
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

/-! ## Unit Tests -/

#eval! countingSort [3, 1, 2]  -- expected: [1, 2, 3]
#eval! countingSort []  -- expected: []
#eval! countingSort [1]  -- expected: [1]
#eval! countingSort [5, 2, 4, 6, 1, 3]  -- expected: [1, 2, 3, 4, 5, 6]
#eval! countingSort [3, 1, 4, 1, 2, 3]  -- expected: [1, 1, 2, 3, 3, 4]

example : countingSort [3, 1, 2] = [1, 2, 3] := by native_decide
example : countingSort [] = [] := by native_decide
example : countingSort [1] = [1] := by native_decide
example : countingSort [5, 2, 4, 6, 1, 3] = [1, 2, 3, 4, 5, 6] := by native_decide
example : countingSort [3, 1, 4, 1, 2, 3] = [1, 1, 2, 3, 3, 4] := by native_decide

/-! ## Theorems -/

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
    print("=" * 80)
    print("Demo: Compiling Lean Code for CountingSort (Task 2)")
    print("=" * 80)
    print()
    
    # Compile the lean code
    print("Compiling Lean code...")
    print()
    
    score, feedback = compile(lean_code)
    
    print(f"Compilation Score: {score}")
    print()
    print("Compilation Feedback:")
    print("-" * 80)
    print(feedback)
    print("-" * 80)
    print()
    
    if score == 1.0:
        print("✓ SUCCESS: Lean code compiled successfully!")
    else:
        print("✗ FAILED: Lean code has compilation errors.")
    print()

if __name__ == "__main__":
    main()

