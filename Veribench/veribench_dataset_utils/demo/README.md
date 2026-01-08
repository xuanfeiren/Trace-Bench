# VeriBench Demo Scripts

This directory contains demo scripts showing how to use the VeriBench evaluation utilities.

## Available Demos

### 1. `compile.py` - Basic Compilation ✅
Demonstrates how to compile Lean 4 code and get compilation feedback.

**Usage:**
```bash
python compile.py
```

**Features:**
- Compiles a sample CountingSort implementation
- Returns compilation score (0.0 or 1.0)
- Provides detailed error feedback if compilation fails

---

### 2. `compile_with_unit_tests.py` - Unit Test Evaluation ✅
Shows how to extract implementation and test it against unit tests from golden reference.

**Usage:**
```bash
python compile_with_unit_tests.py
```

**Features:**
- Extracts implementation from generated code
- Combines with unit tests from golden reference
- Compiles and validates against test cases
- Returns unit test pass/fail score

---

### 3. `evaluate_with_llm_judge.py` - Complete 3-Step Evaluation ⭐ NEW

**Complete evaluation pipeline combining all three metrics: compilation, unit tests, and LLM judge.**

**Usage:**
```bash
cd veribench_dataset_utils/demo
uv run python evaluate_with_llm_judge.py
```

**What it does:**

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Compilation                                         │
│ ✓ Does the generated code compile?                          │
│ Score: 0.0 or 1.0                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │ (if 1.0, continue)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Unit Tests                                          │
│ ✓ Does it pass the golden reference unit tests?             │
│ Score: 0.0 or 1.0                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: LLM Judge                                           │
│ ✓ Is it semantically equivalent to golden reference?        │
│ Score: 0-30 (normalized to 0.0-1.0)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ FINAL SCORE                                                 │
│ 30% compilation + 30% unit tests + 40% LLM judge           │
└─────────────────────────────────────────────────────────────┘
```

**Example Output:**
```
================================================================================
EVALUATION SUMMARY
================================================================================
Step 1 - Compilation:  ✓ PASS (Score: 1.00)
Step 2 - Unit Tests:   ✓ PASS (Score: 1.00)
Step 3 - LLM Judge:    50.0% (Score: 15/30)

FINAL SCORE: 80.00%
================================================================================
```

**Output File:**
The demo creates `evaluation_task_{task_id}_full.json` with detailed results:
```json
{
  "task_id": 2,
  "step1_compilation": {
    "score": 1.0,
    "feedback": "The answer is correct! No need to change anything."
  },
  "step2_unit_tests": {
    "score": 1.0,
    "feedback": "The answer is correct! No need to change anything."
  },
  "step3_llm_judge": {
    "score": 15,
    "normalized_score": 0.5,
    "rationale": "The agent's implementation has significant differences..."
  },
  "final_score": 0.8,
  "passed_all_steps": false
}
```

---

## Usage in Your Code

### Basic Usage: 3-Step Evaluation

```python
from evaluate_with_llm_judge import evaluate_generated_code_full

# Your generated Lean code
generated_code = """
namespace MyImplementation
def myFunction (input : List Nat) : List Nat := 
  -- Your implementation here
  input
end MyImplementation
"""

# Run complete evaluation
results = evaluate_generated_code_full(
    task_id=2,                    # Task ID (0-139)
    generated_code=generated_code, # Generated Lean code
    use_llm_judge=True            # Enable LLM judge (requires API)
)

# Access results
print(f"Compilation: {results['step1_compilation']['score']}")
print(f"Unit Tests: {results['step2_unit_tests']['score']}")
print(f"LLM Judge: {results['step3_llm_judge']['normalized_score']:.2%}")
print(f"Final Score: {results['final_score']:.2%}")
```

### Using Individual Components

```python
# Step 1: Just compile
from eval_utils import compile
score, feedback = compile(generated_code)

# Step 2: Compile with unit tests
from eval_utils import combine_code_with_tests
combined = combine_code_with_tests(task_id, generated_code)
score, feedback = compile(combined)

# Step 3: LLM judge only
from llm_judge_utils import judge_generated_code
result = judge_generated_code(task_id, generated_code)
print(f"Score: {result['score']}/30")
print(f"Rationale: {result['rationale']}")
```

---

## Requirements

- Python 3.11+
- Lean 4 (installed via elan) - for compilation and unit tests
- OpenAI Python library - for LLM judge
- API access to Claude (via OpenAI-compatible endpoint)

**Install dependencies:**
```bash
cd /path/to/Veribench
uv add openai
```

---

## Configuration

Ensure `secrets_local.py` exists in `my_processing_agents/` with:

```python
import os

MODEL = "claude-3.5-sonnet"
os.environ["TRACE_CUSTOMLLM_URL"] = "http://your-endpoint:4000"
os.environ["TRACE_CUSTOMLLM_API_KEY"] = "your-api-key"
os.environ["TRACE_CUSTOMLLM_MODEL"] = "claude-3.5-sonnet"
```

---

## Scoring Breakdown

### Final Score Calculation

**With LLM Judge:**
```
Final Score = 0.3 × Compilation + 0.3 × Unit Tests + 0.4 × LLM Judge
```

**Without LLM Judge:**
```
Final Score = 0.5 × Compilation + 0.5 × Unit Tests
```

### What Each Step Measures

**Step 1: Compilation (30%)**
- ✓ Pass: Code compiles without errors
- ✗ Fail: Syntax errors, type errors, or import errors

**Step 2: Unit Tests (30%)**
- ✓ Pass: Implementation passes all unit tests from golden reference
- ✗ Fail: Wrong output, runtime errors, or logic errors

**Step 3: LLM Judge (40%)**
- Evaluates semantic equivalence on 0-30 scale
- Considers:
  - Correctness of main implementation
  - Quality of helper functions
  - Mathematical rigor (theorems, proofs)
  - Completeness vs golden reference

---

## Output Files

| File | Description |
|------|-------------|
| `evaluation_task_{id}_full.json` | Complete 3-step evaluation results |
| `llm_judge_task_{id}_result.json` | LLM judge only results |

---

## Troubleshooting

### Compilation Errors
- Check Lean 4 installation: `lake env lean --version`
- Verify code syntax matches Lean 4 conventions
- Review error feedback for specific issues

### Unit Test Failures
- Ensure function signatures match golden reference
- Check implementation logic against test cases
- Review combined code to see how tests are integrated

### LLM Judge Issues
- Verify API credentials in `secrets_local.py`
- Check network connectivity
- Ensure endpoint is OpenAI-compatible
- Set `use_llm_judge=False` to skip LLM judge step

---

## Examples

### Example 1: Quick Evaluation
```bash
cd veribench_dataset_utils/demo
uv run python evaluate_with_llm_judge.py
```

### Example 2: Evaluate Your Own Code
```python
from evaluate_with_llm_judge import evaluate_generated_code_full

my_code = """
namespace MyTask
def solution (n : Nat) : Nat := n * 2
end MyTask
"""

results = evaluate_generated_code_full(
    task_id=0,
    generated_code=my_code,
    use_llm_judge=True
)

if results['passed_all_steps']:
    print("🎉 Success!")
else:
    print(f"Score: {results['final_score']:.0%}")
```

### Example 3: Skip LLM Judge (Faster)
```python
results = evaluate_generated_code_full(
    task_id=2,
    generated_code=my_code,
    use_llm_judge=False  # Only compile and unit test
)
```

---

## References

- **Evaluation Utils**: `../eval_utils.py`
- **LLM Judge Utils**: `../llm_judge_utils.py`
- **Main README**: `../../README.md`
- **VeriBench Bundle**: `../../self-opt-data-gen/veribench_bundle/`

