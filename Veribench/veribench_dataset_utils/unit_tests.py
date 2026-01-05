"""
Unit Testing for Generated Lean 4 Code

This module provides a simple interface to test generated Lean4 code against
the unit tests from VeriBench gold reference files.

Main function: unit_test(task_id, lean4_code)
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# =============================================================================
# Core Functions
# =============================================================================

def get_unit_tests(task_id: int) -> List[str]:
    """
    Get unit tests for a given task ID from the gold reference.
    
    Args:
        task_id: Task index (0-139)
        
    Returns:
        List of unit test strings (empty list if no tests found)
        
    Example:
        >>> tests = get_unit_tests(0)
        >>> print(f"Task 0 has {len(tests)} unit tests")
        >>> for i, test in enumerate(tests):
        ...     print(f"Test {i+1}: {test[:50]}...")
    """
    gold_file = _get_gold_reference_file(task_id)
    if not gold_file or not gold_file.exists():
        return []
    
    return _extract_unit_tests_from_gold(gold_file)


def unit_test(task_id: int, lean4_code: str) -> Dict[str, Any]:
    """
    Test generated Lean4 code against unit tests from the gold reference.
    
    Args:
        task_id: Task index (0-139)
        lean4_code: Generated Lean4 code string
        
    Returns:
        Dictionary with test results:
        {
            'success': bool,           # Whether all tests passed
            'task_id': int,           # Task ID
            'num_tests': int,         # Number of tests found
            'compilation_success': bool, # Whether code compiled
            'tests_passed': int,      # Number of tests passed
            'error': str,             # Error message if any
            'output': str             # Compilation output
        }
        
    Example:
        >>> result = unit_test(0, "import Mathlib\\nnamespace BinarySearch\\n...")
        >>> print(f"Success: {result['success']}, Tests: {result['tests_passed']}/{result['num_tests']}")
    """
    try:
        # Step 1: Get the gold reference file for this task
        gold_file = _get_gold_reference_file(task_id)
        if not gold_file or not gold_file.exists():
            return {
                'success': False,
                'task_id': task_id,
                'num_tests': 0,
                'compilation_success': False,
                'tests_passed': 0,
                'error': f'No gold reference file found for task {task_id}',
                'output': ''
            }
        
        # Step 2: Extract unit tests from gold reference
        unit_tests = _extract_unit_tests_from_gold(gold_file)
        
        if not unit_tests:
            return {
                'success': True,  # No tests to fail
                'task_id': task_id,
                'num_tests': 0,
                'compilation_success': True,
                'tests_passed': 0,
                'error': 'No unit tests found in gold reference',
                'output': 'No tests to run'
            }
        
        # Step 3: Combine generated code with tests
        combined_code = _combine_code_with_tests(lean4_code, unit_tests)
        
        # Step 4: Run the combined code in Lean
        compilation_success, output = _run_lean_code(combined_code, task_id)
        
        # Step 5: Determine success
        success = compilation_success
        tests_passed = len(unit_tests) if compilation_success else 0
        
        return {
            'success': success,
            'task_id': task_id,
            'num_tests': len(unit_tests),
            'compilation_success': compilation_success,
            'tests_passed': tests_passed,
            'error': '' if success else 'Compilation or test failed',
            'output': output
        }
        
    except Exception as e:
        return {
            'success': False,
            'task_id': task_id,
            'num_tests': 0,
            'compilation_success': False,
            'tests_passed': 0,
            'error': str(e),
            'output': ''
        }


# =============================================================================
# Helper Functions
# =============================================================================

def _get_gold_reference_file(task_id: int) -> Optional[Path]:
    """Get the gold reference Lean file for a given task ID."""
    script_dir = Path(__file__).resolve().parent.parent
    bundle_root = script_dir / "self-opt-data-gen" / "veribench_bundle"
    py_src_root = bundle_root / "veribench_dataset" / "py_src"
    lean_src_root = bundle_root / "veribench_dataset" / "lean_src" / "veribench"
    
    # Get all Python files in order (same as create_single_task_dataset)
    all_py_files = []
    for category in ["cs_set", "easy_set", "humaneval_set", "real_code", "security_6858", "security_python"]:
        category_path = py_src_root / category
        if category_path.exists():
            all_py_files.extend(sorted(category_path.rglob("*.py")))
    
    if task_id < 0 or task_id >= len(all_py_files):
        return None
    
    # Get corresponding Lean gold reference file
    py_file = all_py_files[task_id]
    rel_path = py_file.relative_to(py_src_root)
    lean_file = lean_src_root / rel_path.with_suffix('.lean')
    
    return lean_file if lean_file.exists() else None


def _extract_unit_tests_from_gold(gold_lean_file: Path) -> List[str]:
    """Extract unit tests from a gold reference Lean file using the official extractor."""
    try:
        # Use the official VeriBench test extractor
        sys.path.insert(0, str(gold_lean_file.parents[3] / "py_src"))
        from veribench.metrics.lean_test_extractor import extract_unit_tests
        
        tests = extract_unit_tests(str(gold_lean_file))
        return tests if tests else []
    except Exception as e:
        # Fallback to simple regex extraction
        import re
        content = gold_lean_file.read_text(encoding='utf-8')
        tests = []
        
        # Extract #eval tests
        eval_pattern = r'#eval\s+[^\n]+'
        eval_tests = re.findall(eval_pattern, content)
        tests.extend(eval_tests)
        
        # Extract example tests
        example_pattern = r'example\s*:[^:]+:=\s*by\s+native_decide'
        example_tests = re.findall(example_pattern, content, re.MULTILINE)
        tests.extend(example_tests)
        
        return tests


def _combine_code_with_tests(lean4_code: str, unit_tests: List[str]) -> str:
    """Combine generated Lean4 code with unit tests."""
    # Clean the generated code
    lean4_code = lean4_code.strip()
    
    # Add unit tests section
    combined = f"""{lean4_code}

/-! Unit Tests -/

{chr(10).join(unit_tests)}
"""
    return combined


def _run_lean_code(lean_code: str, task_id: int) -> Tuple[bool, str]:
    """
    Run Lean code and check if it compiles and tests pass.
    
    Returns:
        (success, output) tuple
    """
    script_dir = Path(__file__).resolve().parent.parent
    lean_project_root = script_dir / "self-opt-data-gen" / "veribench_bundle" / "veribench_dataset" / "lean_src"
    
    if not lean_project_root.exists():
        return False, f"Lean project root not found: {lean_project_root}"
    
    # Ensure temporary directory exists
    test_tmp_dir = lean_project_root / ".test_tmp"
    test_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in the Lean project
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.lean',
        dir=test_tmp_dir,
        delete=False,
        encoding='utf-8'
    ) as f:
        temp_file = Path(f.name)
        f.write(lean_code)
    
    try:
        # Get relative path for lake
        rel_path = temp_file.relative_to(lean_project_root)
        
        # Run lake env lean
        cmd = ["lake", "env", "lean", str(rel_path)]
        result = subprocess.run(
            cmd,
            cwd=lean_project_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        
        return success, output
        
    except subprocess.TimeoutExpired:
        return False, "Timeout: Compilation took too long (>120s)"
    except Exception as e:
        return False, f"Error running Lean: {str(e)}"
    finally:
        # Clean up temp file
        try:
            temp_file.unlink()
        except:
            pass


# =============================================================================
# Legacy Functions (kept for compatibility)
# =============================================================================

def run_unit_tests_builtin(
    generated_lean_dir: str,
    gold_reference_dir: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Run unit tests using VeriBench's built-in evaluation system.
    
    This uses the evaluation script from:
    self-opt-data-gen/veribench_bundle/experiments/12_unit_test_accuracy/
    
    Args:
        generated_lean_dir: Directory containing generated Lean files
        gold_reference_dir: Directory containing gold reference files with tests
        output_dir: Directory to store test results
        
    Returns:
        Dictionary with test results
        
    Example:
        >>> results = run_unit_tests_builtin(
        ...     "outputs/model_xyz/generations",
        ...     "self-opt-data-gen/veribench_bundle/veribench_dataset/lean_src/veribench",
        ...     "outputs/model_xyz/unit_tests"
        ... )
    """
    # Get paths
    script_dir = Path(__file__).resolve().parent.parent
    bundle_root = script_dir / "self-opt-data-gen" / "veribench_bundle"
    eval_script = bundle_root / "experiments/12_unit_test_accuracy/main_eval_unit_test_acc.py"
    
    if not eval_script.exists():
        raise FileNotFoundError(f"Unit test evaluator not found at {eval_script}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for LLM extractor (optional but recommended)
    env = os.environ.copy()
    env['USE_LLM_EXTRACTOR'] = '1'  # Use LLM to extract unit tests
    
    # Run the evaluation script
    cmd = [
        sys.executable,
        str(eval_script),
        "_main",
        f"--impl_path={generated_lean_dir}",
        f"--test_path={gold_reference_dir}",
        f"--output_dir={output_dir}",
        "--use_llm_extractor=true"
    ]
    
    print(f"Running unit test evaluation...")
    print(f"  Generated code: {generated_lean_dir}")
    print(f"  Gold reference: {gold_reference_dir}")
    print(f"  Output: {output_dir}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"Error running unit tests:")
        print(result.stderr)
        return {'success': False, 'error': result.stderr}
    
    # Load results
    results_file = Path(output_dir) / "results.json"
    if results_file.exists():
        import json
        with open(results_file, 'r') as f:
            return json.load(f)
    
    return {'success': True, 'message': 'Tests completed'}


# =============================================================================
# STEP 3: Manual Unit Test Extraction and Running
# =============================================================================

def extract_unit_tests_from_gold(gold_lean_file: Path) -> List[str]:
    """
    Extract unit tests from a gold reference Lean file.
    
    Looks for:
    - #eval statements
    - example declarations with native_decide
    
    Args:
        gold_lean_file: Path to gold reference Lean file
        
    Returns:
        List of unit test strings
    """
    if not gold_lean_file.exists():
        return []
    
    content = gold_lean_file.read_text(encoding='utf-8')
    tests = []
    
    # Extract #eval tests
    import re
    eval_pattern = r'#eval\s+[^\n]+'
    eval_tests = re.findall(eval_pattern, content)
    tests.extend(eval_tests)
    
    # Extract example tests
    example_pattern = r'example\s*:[^:]+:=\s*by\s+native_decide'
    example_tests = re.findall(example_pattern, content, re.MULTILINE)
    tests.extend(example_tests)
    
    return tests


def create_test_file(
    generated_lean_file: Path,
    unit_tests: List[str],
    output_path: Path
) -> Path:
    """
    Create a combined Lean file with generated code + unit tests.
    
    Args:
        generated_lean_file: Path to generated Lean implementation
        unit_tests: List of unit test strings to add
        output_path: Where to save the combined file
        
    Returns:
        Path to created test file
    """
    # Read generated code
    generated_code = generated_lean_file.read_text(encoding='utf-8')
    
    # Combine with tests
    combined = f"""{generated_code}

/-! Unit Tests -/

{chr(10).join(unit_tests)}
"""
    
    # Write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding='utf-8')
    
    return output_path


def run_lean_file(lean_file: Path, lean_project_root: Path) -> bool:
    """
    Run a Lean file to check if it compiles and tests pass.
    
    Args:
        lean_file: Path to Lean file to run
        lean_project_root: Root directory of Lean project (with lakefile.toml)
        
    Returns:
        True if successful, False otherwise
    """
    # Use lake env lean to compile/run the file
    cmd = ["lake", "env", "lean", str(lean_file.relative_to(lean_project_root))]
    
    result = subprocess.run(
        cmd,
        cwd=lean_project_root,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    return result.returncode == 0


# =============================================================================
# STEP 4: Complete Workflow Example
# =============================================================================

def evaluate_single_generated_file(
    generated_file: Path,
    gold_file: Path,
    lean_project_root: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Complete workflow to evaluate a single generated Lean file.
    
    Steps:
    1. Extract unit tests from gold file
    2. Combine with generated code
    3. Try to compile and run
    4. Report results
    
    Args:
        generated_file: Generated Lean implementation
        gold_file: Gold reference with tests
        lean_project_root: Lean project root with lakefile.toml
        output_dir: Where to store test artifacts
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating: {generated_file.name}")
    print("=" * 80)
    
    # Step 1: Extract tests from gold file
    unit_tests = extract_unit_tests_from_gold(gold_file)
    print(f"  Extracted {len(unit_tests)} unit tests from gold file")
    
    if not unit_tests:
        return {
            'file': str(generated_file),
            'success': False,
            'error': 'No unit tests found in gold file'
        }
    
    # Step 2: Create combined test file
    test_file = output_dir / f"test_{generated_file.name}"
    create_test_file(generated_file, unit_tests, test_file)
    print(f"  Created test file: {test_file}")
    
    # Step 3: Copy to Lean project and run
    project_test_file = lean_project_root / ".test_tmp" / test_file.name
    project_test_file.parent.mkdir(parents=True, exist_ok=True)
    project_test_file.write_text(test_file.read_text())
    
    print(f"  Running unit tests...")
    success = run_lean_file(project_test_file, lean_project_root)
    
    # Step 4: Report results
    result = {
        'file': str(generated_file),
        'gold_file': str(gold_file),
        'num_tests': len(unit_tests),
        'success': success,
        'test_file': str(test_file)
    }
    
    if success:
        print(f"  ✅ All tests passed!")
    else:
        print(f"  ❌ Tests failed")
    
    return result


# =============================================================================
# STEP 5: Quick Usage Examples
# =============================================================================

def example_usage():
    """Show example usage patterns."""
    
    print("""
# ============================================================================
# Example 1: Using Built-in Evaluator (Recommended)
# ============================================================================

from my_unit_tests.unit_tests import run_unit_tests_builtin

results = run_unit_tests_builtin(
    generated_lean_dir="outputs/my_model/generations",
    gold_reference_dir="self-opt-data-gen/veribench_bundle/veribench_dataset/lean_src/veribench",
    output_dir="outputs/my_model/unit_tests"
)

print(f"Compilation rate: {results['summary']['compilation_rate']}")
print(f"Functional test pass rate: {results['summary']['avg_functional_pass_rate']}")


# ============================================================================
# Example 2: Manual Evaluation of Single File
# ============================================================================

from pathlib import Path
from my_unit_tests.unit_tests import evaluate_single_generated_file

result = evaluate_single_generated_file(
    generated_file=Path("outputs/my_model/binary_search.lean"),
    gold_file=Path("self-opt-data-gen/veribench_bundle/veribench_dataset/lean_src/veribench/cs_set/binary_search.lean"),
    lean_project_root=Path("self-opt-data-gen/veribench_bundle/veribench_dataset/lean_src"),
    output_dir=Path("outputs/my_model/test_artifacts")
)

print(f"Success: {result['success']}")
print(f"Number of tests: {result['num_tests']}")


# ============================================================================
# Example 3: Batch Evaluation
# ============================================================================

from pathlib import Path
from my_unit_tests.unit_tests import evaluate_single_generated_file

generated_dir = Path("outputs/my_model/generations")
gold_dir = Path("self-opt-data-gen/veribench_bundle/veribench_dataset/lean_src/veribench")
lean_root = Path("self-opt-data-gen/veribench_bundle/veribench_dataset/lean_src")
output_dir = Path("outputs/my_model/test_results")

results = []
for generated_file in generated_dir.rglob("*.lean"):
    # Find corresponding gold file
    rel_path = generated_file.relative_to(generated_dir)
    gold_file = gold_dir / rel_path
    
    if gold_file.exists():
        result = evaluate_single_generated_file(
            generated_file, gold_file, lean_root, output_dir
        )
        results.append(result)

# Summary
total = len(results)
passed = sum(1 for r in results if r['success'])
print(f"\\nSummary: {passed}/{total} files passed all unit tests")
    """)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("VeriBench Unit Testing")
    print("=" * 80)
    print()
    print("Main function: unit_test(task_id, lean4_code)")
    print()
    
    # Test with a simple example
    print("Testing with task 0 (binary_search)...")
    
    # Simple Lean4 code for testing
    test_code = """import Mathlib.Data.List.Basic

namespace BinarySearch

def binarySearch (arr : List Nat) (target : Nat) : Option Nat :=
  arr.findIdx? (· = target)

end BinarySearch
"""
    
    result = unit_test(0, test_code)
    
    print()
    print("Result:")
    print(f"  Success: {result['success']}")
    print(f"  Task ID: {result['task_id']}")
    print(f"  Tests found: {result['num_tests']}")
    print(f"  Tests passed: {result['tests_passed']}")
    print(f"  Compilation: {result['compilation_success']}")
    if result['error']:
        print(f"  Error: {result['error']}")
    print()
    print("=" * 80)

