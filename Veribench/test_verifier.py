#!/usr/bin/env python3
"""
Test script to demonstrate the Veribench verifier working with agent-generated solutions.
"""

from verifier import VeribenchVerifier
from datasets import load_dataset

def test_verifier_with_sample_solution():
    """Test the verifier with a sample Lean 4 solution."""
    
    # Sample Lean 4 solution (based on the example structure from Veribench)
    sample_solution = '''
```lean
/-!
# VeriBench – Binary Search

Binary search implementation that searches for a target value in a sorted list.
Returns the index if found, None if not found.

## Implementation

Defines a binary search function with preconditions and correctness theorems.
-/

namespace BinarySearch

/--
**Implementation of binary search.**

`binarySearch arr target` returns the index of target in sorted array arr, or none if not found.

## Examples

#eval binarySearch [1, 2, 3, 4, 5] 3 -- expected: some 2
#eval binarySearch [1, 2, 3, 4, 5] 6 -- expected: none
-/
def binarySearch (arr : List Int) (target : Int) : Option Nat := 
  sorry -- Implementation would go here

/-!
# Tests
-/

/-- Basic functionality test -/
example : binarySearch [1, 2, 3, 4, 5] 3 = some 2 := by sorry
#eval binarySearch [1, 2, 3, 4, 5] 3 -- expected: some 2

/-- Not found test -/
example : binarySearch [1, 2, 3, 4, 5] 6 = none := by sorry
#eval binarySearch [1, 2, 3, 4, 5] 6 -- expected: none

/-!
# Pre-Condition
-/

/-- **Pre-condition.** Array is sorted and contains only integers. -/
def Pre (arr : List Int) (target : Int) : Prop := 
  List.Sorted (· ≤ ·) arr

/-!
# Correctness Theorem
-/

/-- **Correctness theorem**: if result is some index, then arr[index] = target -/
theorem correctness_thm (arr : List Int) (target : Int) (hPre : Pre arr target) :
  ∀ idx, binarySearch arr target = some idx → arr.get? idx = some target := by
  sorry

end BinarySearch
```
'''
    
    print("Testing Veribench Verifier")
    print("=" * 50)
    
    # Initialize verifier
    verifier = VeribenchVerifier(verbose=True)
    
    # Test with the sample solution
    result = verifier.verify_solution(sample_solution, task_index=0)
    
    print(f"\nVerification completed!")
    print(f"Success: {result['success']}")
    print(f"Overall Score: {result['overall_score']:.2f}")
    
    return result

def test_verifier_with_invalid_solution():
    """Test the verifier with an invalid solution."""
    
    invalid_solution = '''
This is not a valid Lean 4 solution.
It's just plain text without proper structure.
'''
    
    print("\n" + "=" * 50)
    print("Testing with invalid solution...")
    
    verifier = VeribenchVerifier(verbose=True)
    result = verifier.verify_solution(invalid_solution, task_index=0)
    
    print(f"Invalid solution test completed!")
    print(f"Success: {result['success']} (should be False)")
    
    return result

if __name__ == "__main__":
    # Test with valid solution
    test_verifier_with_sample_solution()
    
    # Test with invalid solution  
    test_verifier_with_invalid_solution()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
