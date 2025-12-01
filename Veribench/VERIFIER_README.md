# Veribench Verifier

A comprehensive verification system for Lean 4 solutions generated for Veribench tasks.

## Overview

The Veribench verifier validates generated Lean 4 code against the expected structure and syntax requirements of the Veribench benchmark. It provides detailed feedback on correctness, completeness, and potential issues.

## Files

- **`verifier.py`** - Main verifier class with comprehensive validation
- **`agent_with_verification.py`** - Integration script combining agent generation with verification
- **`test_verifier.py`** - Test script demonstrating verifier functionality

## Features

### Structure Validation
- ✅ **Docstring blocks** (`/-! ... -/`)
- ✅ **Function definitions** (`def ...`)
- ✅ **Unit tests** (`#eval`, `example`)
- ✅ **Theorems/Lemmas** (`theorem`, `lemma`)
- ✅ **Namespace organization**

### Syntax Validation
- ✅ **PyPantograph integration** for Lean 4 syntax checking
- ✅ **Basic pattern matching** for common syntax errors
- ✅ **Bracket matching** validation
- ✅ **Graceful fallback** when PyPantograph unavailable

### Comprehensive Reporting
- ✅ **Detailed feedback** on structure and syntax issues
- ✅ **Scoring system** (0.0 to 1.0) for solution quality
- ✅ **JSON output** support for automation
- ✅ **Verbose and quiet modes**

## Usage

### Basic Verification

```bash
# Verify a solution from file
python verifier.py --task_index 0 --solution_file solution.lean

# Verify a solution from string
python verifier.py --task_index 0 --solution "lean code here"

# Get JSON output
python verifier.py --task_index 0 --solution_file solution.lean --json_output
```

### Agent + Verifier Pipeline

```bash
# Generate and verify a single task
python agent_with_verification.py --task_index 0

# Run batch evaluation on multiple tasks
python agent_with_verification.py --batch --start_index 0 --num_tasks 5

# Generate without saving files
python agent_with_verification.py --task_index 0 --no_save
```

### Testing

```bash
# Run verifier tests
python test_verifier.py
```

## Dependencies

### Required
- `datasets` - For loading Veribench dataset
- `numpy`, `pandas` - Data processing
- `litellm` - LLM API calls (for agent integration)

### Optional
- `pantograph` - For advanced Lean 4 syntax validation
  ```bash
  uv sync --extra lean4
  ```

## Verification Criteria

### Structure Score (5 points max)
1. **Docstring block** (1 point) - Proper `/-! ... -/` documentation
2. **Function definitions** (1 point) - At least one `def` statement
3. **Unit tests** (1 point) - `#eval` or `example` statements
4. **Theorems** (1 point) - `theorem` or `lemma` declarations
5. **Namespace** (1 point) - Proper namespace organization

### Syntax Score
- **Valid syntax** - Code passes basic Lean 4 syntax checks
- **PyPantograph validation** - Advanced syntax verification when available

### Overall Score Calculation
```
Overall Score = 0.6 × (Structure Score / 5) + 0.4 × Syntax Score
```

## Example Output

```
==============================================================
VERIBENCH SOLUTION VERIFICATION
==============================================================
✓ Loaded task: binary_search.py (category: cs_set)
✓ Extracted Lean code (1234 characters)

Task: binary_search.py (Category: cs_set)
Overall Score: 0.85/1.00
Success: ✓

📋 STRUCTURE VALIDATION:
  Docstring block: ✓
  Function definitions: ✓
  Unit tests: ✓
  Theorems: ✓
  Namespace: ✓
  Structure Score: 5/5

🔍 SYNTAX VALIDATION:
  Valid syntax: ✓
  PyPantograph available: ✓
==============================================================
```

## Integration with Trace-Bench

The verifier is designed to integrate seamlessly with the broader Trace-Bench ecosystem:

- **Dataset compatibility** - Works with `allenanie/veribench_with_prompts`
- **PyPantograph integration** - Leverages existing Lean 4 infrastructure
- **Modular design** - Can be used standalone or with agents
- **Extensible validation** - Easy to add new verification criteria

## Future Enhancements

- **Semantic validation** - Check logical correctness of proofs
- **Performance benchmarking** - Measure compilation and execution time
- **Automated theorem proving** - Verify proof completeness
- **Multi-language support** - Extend to other formal languages
- **Interactive feedback** - Suggest fixes for common issues

## Error Handling

The verifier gracefully handles various error conditions:

- **Missing PyPantograph** - Falls back to basic syntax checking
- **Invalid Lean code** - Provides detailed error messages
- **Network issues** - Caches dataset locally when possible
- **File I/O errors** - Clear error reporting and recovery

## Contributing

To extend the verifier:

1. Add new validation criteria in `validate_lean_structure()`
2. Enhance syntax checking in `check_lean_syntax()`
3. Update scoring in `_calculate_overall_score()`
4. Add tests in `test_verifier.py`

The verifier is designed to be modular and extensible for future Veribench requirements.
