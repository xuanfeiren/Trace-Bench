"""
This file contains utility functions for evaluating the performance of the model.
"""

import sys
import tempfile
import logging
from pathlib import Path
import requests

# Suppress HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Setup paths for imports
_current_file = Path(__file__).resolve()
_veribench_dataset_utils = _current_file.parent
_project_root = _veribench_dataset_utils.parent

# Add project root to path for imports
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root / "my_processing_agents") not in sys.path:
    sys.path.insert(0, str(_project_root / "my_processing_agents"))

# Now import
from my_processing_agents.lean_interpretor import lean_interpreter

# Try to import print_color, fall back to print if not available
from opto.optimizers.utils import print_color

# Setup VeriBench paths for unit test integration (done at module level)
_veribench_root = Path(__file__).resolve().parent.parent / "self-opt-data-gen" / "veribench_bundle"
_py_src = _veribench_root / "veribench_dataset" / "py_src"
_experiments_path = _veribench_root / "experiments" / "12_unit_test_accuracy"

if str(_py_src) not in sys.path:
    sys.path.insert(0, str(_py_src))
if str(_experiments_path) not in sys.path:
    sys.path.insert(0, str(_experiments_path))

# VeriBench utilities will be imported lazily in functions to avoid dependency issues

def remove_import_error(response,result):
    """Remove the import error from the response."""
    error_messages = result["error_messages"]
    has_import_errors = any("invalid 'import' command" in msg for msg in error_messages)

    if has_import_errors:
        # Remove import error lines
        code_lines = response.strip().split('\n')
        problematic_lines = set()
        
        for error_msg in error_messages:
            if "invalid 'import' command" in error_msg:
                try:
                    parts = error_msg.split(':')
                    if len(parts) >= 3:
                        line_num = int(parts[1])
                        if 1 <= line_num <= len(code_lines):
                            problematic_lines.add(line_num - 1)
                except Exception:
                    pass
        
        # Remove problematic import lines and re-evaluate
        if problematic_lines:
            filtered_lines = [line for i, line in enumerate(code_lines) if i not in problematic_lines]
            response = '\n'.join(filtered_lines)
    return response

def compile(response):
    """Compile the lean code, return the score and feedback.
    
    Args:
        task: The task being evaluated (user query)
        response: The LLM-generated Lean code response
        
    Returns:
        Tuple of (score, feedback)
        The score and feedback only contains the information about the compilation error.
    """
    try:
        result = lean_interpreter(response)
        correctness = result["valid"]
        score = 1.0 if correctness else 0.0

        if correctness:
            feedback = "The answer is correct! No need to change anything."
            return score, feedback
        
        num_errors = result["num_errors"]
        
        # Get error details with context if available, otherwise raw messages
        error_details = result.get("error_details", result["error_messages"])
        errors_str = "\n\n".join(error_details)
        
        # Build feedback
        feedback = f"Lean compilation FAILED with {num_errors} error(s). Please make the Lean code correct and as simple as possible.\n\nErrors:\n{errors_str}"

        if 'TimeoutError' in errors_str:
            # Time out error is not a simple Lean compilation error, so we return 0.0.
            print_color("Lean code compilation TIMEOUT. Return timeout feedback.", "yellow")
            return 0.0, "Lean code compilation TIMEOUT. The generated code is either incorrect or too complex for the interpreter to compile within the time limit. Please make the Lean code correct and as simple as possible."
        
        # I assume there will not be timeout error after removing import errors...

        cleaned_code = remove_import_error(response,result)
        if cleaned_code != response:
            # Re-run interpreter on cleaned code
            result = lean_interpreter(cleaned_code)
            # assert current no import errors
            while any("invalid 'import' command" in msg for msg in result["error_messages"]):
                cleaned_code = remove_import_error(cleaned_code,result)
                result = lean_interpreter(cleaned_code)

            assert not any("invalid 'import' command" in msg for msg in result["error_messages"]), "There are still import errors after removing import errors."
        
            correctness = result["valid"]
            score = 1.0 if correctness else 0.0
        
            # Update feedback based on new result
            if correctness:
                feedback = "The answer is correct after removing invalid import statements!"
            else:
                num_errors = result["num_errors"]
                error_details = result.get("error_details", result["error_messages"])
                errors_str = "\n\n".join(error_details)
                feedback = f"Lean compilation FAILED with {num_errors} error(s) after removing invalid imports. Please make the Lean code correct and as simple as possible.\n\nErrors:\n{errors_str}"

        return score, feedback

    except Exception as e:
        print_color(f"Error: {e}", "red")
        raise e
        

def combine_code_with_tests(task_id: int, lean_code: str) -> str:
    """
    Combine the lean code with the unit tests from gold reference.
    
    This function:
    1. Extracts implementation from generated lean_code
    2. Extracts unit tests from gold reference
    3. Automatically matches function names and creates aliases
    4. Combines them into a compilable Lean file
    
    Args:
        task_id: Task ID (0-139)
        lean_code: Generated Lean code
        
    Returns:
        Combined Lean code with matched unit tests
    """
    # Lazy import to avoid dependency issues
    from veribench.metrics.lean_test_extractor import extract_unit_tests as extract_tests
    from main_eval_unit_test_acc import create_combined_file
    import re
    import logging
    
    # Suppress INFO logging from lean_test_extractor
    logging.getLogger('veribench.metrics.lean_test_extractor').setLevel(logging.WARNING)
    
    # Alternative: Load from our local dataset JSON to avoid fastchat dependency
    dataset_root = Path(__file__).resolve().parent.parent / "veribench_dataset_utils" / "dataset"
    task_file = dataset_root / f"task_{task_id}.json"
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    import json
    with open(task_file, 'r') as f:
        task_data = json.load(f)
    
    # Use gold reference code directly from our dataset
    gold_lean_code = task_data.get('gold_reference_lean4_code', '')
    if not gold_lean_code:
        raise ValueError(f"No gold reference Lean code found for task {task_id}")
    
    # Save gold reference to temp file for extraction
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False, encoding='utf-8') as temp_gold:
        temp_gold.write(gold_lean_code)
        temp_gold_path = temp_gold.name
    
    gold_lean_file = Path(temp_gold_path)
    
    try:
        # More robust extraction: Extract all function definitions from generated code
        # This handles multi-line definitions and various formats better
        implementation = _extract_implementation_robust(lean_code)
        
        # Extract unit tests from gold reference
        unit_tests = extract_tests(str(gold_lean_file))
        
        # Use VeriBench's smart combining function
        # This automatically:
        # - Matches function names between implementation and tests
        # - Creates aliases (abbrev) to bridge naming differences
        # - Handles namespaces correctly
        # - Reorders definitions to resolve dependencies
        combined_code = create_combined_file(
            implementation=implementation,
            tests=unit_tests,
            imperative_impl="",  # No imperative implementation for now
            imperative_tests=[],
            disable_aliases=False,  # Enable automatic name matching
            keep_tests_exact=False,  # Allow test modification
            let_llm_order=False  # Use smart reordering
        )
        
        return combined_code
        
    finally:
        # Clean up temp file
        try:
            gold_lean_file.unlink()
        except:
            pass


def _extract_implementation_robust(lean_code: str) -> str:
    """
    Robust extraction of implementation code from generated Lean code.
    Extracts imports, namespaces, and all function definitions.
    Stops at test/example sections.
    """
    import re
    
    lines = lean_code.split('\n')
    extracted = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Stop at test/example sections
        if (re.match(r'^\s*/-!\s*#?\s*(Unit\s*)?Tests', line, re.IGNORECASE) or
            re.match(r'^\s*--\s*Unit\s*[Tt]ests', line) or
            re.match(r'^\s*example\s*:', line) or
            re.match(r'^\s*#eval\s+', line)):
            break
        
        # Skip comments and empty lines unless they're doc comments for functions
        if not stripped or stripped.startswith('--') or stripped.startswith('#'):
            i += 1
            continue
        
        # Skip block comments (but keep doc comments)
        if stripped.startswith('/-') and not stripped.startswith('/-!'):
            # Skip until closing -/
            while i < len(lines) and '-/' not in lines[i]:
                i += 1
            i += 1
            continue
        
        # Capture imports, namespaces, opens
        if (stripped.startswith('import ') or 
            stripped.startswith('namespace ') or 
            stripped.startswith('open ') or
            stripped.startswith('variable ') or
            stripped.startswith('universe ')):
            extracted.append(line)
            i += 1
            continue
        
        # Capture function definitions (def, partial def, noncomputable def, abbrev)
        if re.match(r'^\s*(partial\s+|noncomputable\s+)?(def|abbrev)\s+\w+', line):
            # Start capturing the function
            func_lines = [line]
            i += 1
            
            # Continue capturing until we find the end of the function
            # A function ends when we hit a blank line followed by a non-indented line,
            # or another top-level definition
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                
                # Stop if we hit another definition or test section
                if (re.match(r'^\s*(partial\s+|noncomputable\s+)?(def|abbrev|theorem|example)\s+', next_line) or
                    re.match(r'^\s*/-!\s*#?\s*(Unit\s*)?Tests', next_line, re.IGNORECASE) or
                    re.match(r'^\s*#eval\s+', next_line)):
                    break
                
                # Include indented lines and empty lines within the function
                if next_stripped or next_line.startswith('  ') or next_line.startswith('\t'):
                    func_lines.append(next_line)
                    i += 1
                else:
                    # Empty line - check if next non-empty line is indented
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    
                    if j < len(lines) and (lines[j].startswith('  ') or lines[j].startswith('\t')):
                        # Still part of the function
                        func_lines.append(next_line)
                        i += 1
                    else:
                        # End of function
                        break
            
            extracted.extend(func_lines)
            continue
        
        i += 1
    
    return '\n'.join(extracted) 


from llm_judge_utils import judge_generated_code

def evaluate(task_id: int, lean_code: str) -> tuple:
    """
    Evaluate the generated Lean code using 3-step pipeline.
    
    Steps:
    1. Compile the generated code
    2. Run unit tests from golden reference
    3. Use LLM judge to score semantic equivalence
    
    Args:
        task_id: Task ID (0-139)
        lean_code: Generated Lean code
        
    Returns:
        tuple: (final_score, feedback)
            - final_score: float (0.3*compile + 0.3*unit_tests + 0.4*llm_judge)
            - feedback: str (descriptive feedback about the evaluation)
    """
    # step 1: compile
    compilation_score, compilation_feedback = compile(lean_code)
    if compilation_score == 1.0:
        # print_color("Lean code compiled successfully.", "green")
        pass
    else:
        # print_color("Lean code compilation failed.", "red")
        # print_color(compilation_feedback, "red")
        feedback = f"The original code failed to compile. Feedback from the compilation: {compilation_feedback}"
        return compilation_score, feedback
    
    # step 2: unit tests
    lean_code_with_unit_tests = combine_code_with_tests(task_id, lean_code)
    # print_color("Lean code with unit tests: ", "green")
    # print_color(lean_code_with_unit_tests, "yellow")
    unit_tests_score, unit_tests_feedback = compile(lean_code_with_unit_tests)
    if unit_tests_score == 1.0:
        pass
    else:
        # print_color("Unit tests failed.", "red")
        # print_color(unit_tests_feedback, "red")
        unit_tests_feedback = f"The Lean code compiled successfully, but it failed the unit tests from the golden reference. This means the implementation logic is incorrect or incomplete. Please fix the code to match the expected behavior.\n\nUnit test compilation errors:\n{unit_tests_feedback}"
        # pass compilation but not unit tests, the score should be 0.3
        return 0.3, unit_tests_feedback
    
    # step 3: LLM judge
    # Note: Pass the ORIGINAL lean_code, not the combined code with unit tests
    # The LLM judge should compare the generated implementation vs golden reference
    results = judge_generated_code(task_id, lean_code, max_score=30)
    # extract the normalized score and rationale
    LLM_judge_score = results['normalized_score']
    LLM_judge_rationale = results['rationale']

    feedback = f"The original code passed compilation and unit tests. We use a LLM judge to score the semantic equivalence between the golden reference and the generated Lean 4 code. Rationale from the LLM judge: {LLM_judge_rationale}"

    final_score = 0.3 * compilation_score + 0.3 * unit_tests_score + 0.4 * LLM_judge_score

    return final_score, feedback

def main():
    """
    Main function to evaluate the generated Lean code.
    """
    task_id = 2
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
    final_score, feedback = evaluate(task_id, generated_code)
    print(f"Final score: {final_score}")
    print(f"Feedback: {feedback}")

if __name__ == "__main__":
    main()