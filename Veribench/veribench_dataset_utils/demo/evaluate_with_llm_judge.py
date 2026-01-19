"""
Demo: 3-Step Evaluation with LLM Judge

This demo shows the complete evaluation pipeline:
1. Compile the generated Lean code
2. Run unit tests against golden reference
3. Use LLM judge to score semantic equivalence

Usage:
    cd veribench_dataset_utils/demo
    uv run python evaluate_with_llm_judge.py
"""

import sys
import json
import logging
from pathlib import Path

# Suppress HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Add parent directory (veribench_dataset_utils) to path
parent_dir = Path(__file__).resolve().parent.parent
project_root = parent_dir.parent

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_root))

from eval_utils import compile, combine_code_with_tests
from llm_judge_utils import judge_generated_code


def evaluate_generated_code_full(task_id: int, generated_code: str, use_llm_judge: bool = True) -> dict:
    """
    Complete 3-step evaluation of generated Lean code.
    
    Step 1: Compilation - Does the code compile?
    Step 2: Unit Tests - Does it pass the golden reference unit tests?
    Step 3: LLM Judge - Is it semantically equivalent to golden reference?
    
    Args:
        task_id: Task ID (0-139)
        generated_code: Generated Lean code string
        use_llm_judge: Whether to use LLM judge (requires API access)
        
    Returns:
        dict with complete evaluation results
    """
    print("=" * 80)
    print(f"3-Step Evaluation for Task {task_id}")
    print("=" * 80)
    print()
    
    results = {
        'task_id': task_id,
        'step1_compilation': {
            'score': 0.0,
            'feedback': ''
        },
        'step2_unit_tests': {
            'score': 0.0,
            'feedback': ''
        },
        'step3_llm_judge': None,
        'final_score': 0.0,
        'passed_all_steps': False
    }
    
    # ========================================================================
    # STEP 1: Compilation
    # ========================================================================
    print("STEP 1: Compiling generated code...")
    print("-" * 80)
    
    compile_score, compile_feedback = compile(generated_code)
    results['step1_compilation']['score'] = compile_score
    results['step1_compilation']['feedback'] = compile_feedback[:500] + ("..." if len(compile_feedback) > 500 else "")
    
    if compile_score == 1.0:
        print("✓ SUCCESS: Code compiles successfully")
    else:
        print("✗ FAILED: Code does not compile")
        print(f"  Error: {compile_feedback[:200]}...")
    print()
    
    if compile_score == 0.0:
        print("Cannot proceed to unit tests - compilation failed.")
        print()
        return results
    
    # ========================================================================
    # STEP 2: Unit Tests
    # ========================================================================
    print("STEP 2: Running unit tests...")
    print("-" * 80)
    
    try:
        combined_code = combine_code_with_tests(task_id, generated_code)
        unit_test_score, unit_test_feedback = compile(combined_code)
        
        results['step2_unit_tests']['score'] = unit_test_score
        results['step2_unit_tests']['feedback'] = unit_test_feedback[:500] + ("..." if len(unit_test_feedback) > 500 else "")
        
        if unit_test_score == 1.0:
            print("✓ SUCCESS: All unit tests passed")
        else:
            print("✗ FAILED: Unit tests failed")
            print(f"  Error: {unit_test_feedback[:200]}...")
        print()
        
    except Exception as e:
        print(f"✗ ERROR: Unit test evaluation failed: {e}")
        results['step2_unit_tests']['feedback'] = f"Error: {str(e)}"
        print()
    
    # ========================================================================
    # STEP 3: LLM Judge (optional, requires API)
    # ========================================================================
    if use_llm_judge:
        print("STEP 3: LLM Judge - Semantic Equivalence Scoring...")
        print("-" * 80)
        
        try:
            llm_result = judge_generated_code(
                task_id=task_id,
                lean_code_implementation=generated_code,
                max_score=30
            )
            
            results['step3_llm_judge'] = {
                'score': llm_result['score'],
                'normalized_score': llm_result['normalized_score'],
                'rationale': llm_result['rationale']
            }
            
            print(f"✓ LLM Judge Score: {llm_result['score']}/30 ({llm_result['normalized_score']:.1%})")
            print()
            print("Rationale:")
            print(llm_result['rationale'])
            print()
            
        except Exception as e:
            print(f"⚠️  LLM Judge failed: {e}")
            results['step3_llm_judge'] = {
                'score': 0,
                'normalized_score': 0.0,
                'rationale': f"Error: {str(e)}"
            }
            print()
    else:
        print("STEP 3: LLM Judge - SKIPPED (use_llm_judge=False)")
        print()
    
    # ========================================================================
    # Calculate Final Score
    # ========================================================================
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    compilation_score = results['step1_compilation']['score']
    unit_test_score = results['step2_unit_tests']['score']
    
    print(f"Step 1 - Compilation:  {'✓ PASS' if compilation_score == 1.0 else '✗ FAIL'} (Score: {compilation_score:.2f})")
    print(f"Step 2 - Unit Tests:   {'✓ PASS' if unit_test_score == 1.0 else '✗ FAIL'} (Score: {unit_test_score:.2f})")
    
    if results['step3_llm_judge']:
        llm_score = results['step3_llm_judge']['normalized_score']
        print(f"Step 3 - LLM Judge:    {llm_score:.1%} (Score: {results['step3_llm_judge']['score']}/30)")
        print()
        print("LLM Judge Rationale:")
        print("-" * 80)
        print(results['step3_llm_judge']['rationale'])
        print("-" * 80)
        
        # Final score: 30% compilation + 30% unit tests + 40% LLM judge
        results['final_score'] = (
            0.3 * compilation_score +
            0.3 * unit_test_score +
            0.4 * llm_score
        )
    else:
        # Without LLM judge: 50% compilation + 50% unit tests
        results['final_score'] = (
            0.5 * compilation_score +
            0.5 * unit_test_score
        )
    
    print()
    print(f"FINAL SCORE: {results['final_score']:.2%}")
    print()
    
    # Check if passed all steps
    results['passed_all_steps'] = (
        compilation_score == 1.0 and
        unit_test_score == 1.0 and
        (results['step3_llm_judge'] is None or results['step3_llm_judge']['normalized_score'] >= 0.8)
    )
    
    if results['passed_all_steps']:
        print("🎉 OVERALL: PASSED ALL STEPS")
    else:
        print("⚠️  OVERALL: Some steps failed or need improvement")
    
    print("=" * 80)
    print()
    
    return results


def main():
    """Demo with example generated code for Task 2 (CountingSort)"""
    
    task_id = 2
    
    # Example generated code (simulating LLM output)
    generated_code = """/-!
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
      let cnt := count[i]!
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
    
    print("Generated Code Preview:")
    print("-" * 80)
    print(generated_code[:300] + "...")
    print("-" * 80)
    print()
    
    # Run the 3-step evaluation
    results = evaluate_generated_code_full(
        task_id=task_id,
        generated_code=generated_code,
        use_llm_judge=True  # Set to False to skip LLM judge
    )
    
    # Save detailed results
    output_file = Path(__file__).parent / f"evaluation_task_{task_id}_full.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()

