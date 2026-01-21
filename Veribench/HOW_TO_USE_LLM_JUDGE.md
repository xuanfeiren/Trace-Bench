# How to Use LLM Judge to Score Generated Lean Code

## Overview

After your generated Lean code **compiles** and **passes unit tests**, you can use an **LLM judge** to evaluate its **semantic equivalence** to the golden reference implementation. This provides a qualitative score (0-10 or 0-30) based on how well the implementation matches the gold standard.

## 📊 Scoring Flow

```
Generated Lean Code
    ↓
1. ✅ Compilation Check (binary: pass/fail)
    ↓
2. ✅ Unit Tests (binary: pass/fail)
    ↓
3. 📝 LLM Judge (score: 0-10 or 0-30)
    → Evaluates semantic equivalence
    → Provides detailed rationale
```

## 🎯 When to Use LLM Judge

Use LLM judge scoring when:
- ✅ Your code **compiles successfully**
- ✅ Your code **passes all unit tests**
- 🎯 You want to evaluate **semantic equivalence** with the gold reference
- 🎯 You want a **qualitative assessment** of code quality

**Note:** LLM judge complements, but does not replace, compilation and unit testing.

## 📁 Key Files

### 1. LLM Judge Implementation
```
self-opt-data-gen/veribench_bundle/veribench_dataset/py_src/veribench/metrics/
├── sim_file_eq_llm_judge.py     # Main LLM judge implementation
├── sim_thm_eq_llm_judge.py      # Theorem equivalence judge
└── utils.py                      # LLM calling utilities
```

### 2. Prompt Templates
```
self-opt-data-gen/veribench_bundle/veribench_prompts/eval_prompts/
├── file_equiv_prompts/
│   ├── simplest_prompt.txt                      # Simple 0-10 scale
│   ├── simplest_prompt_varying_score_range.txt  # Configurable scale (0-N)
│   ├── file_eq_prompt.txt                       # Detailed prompt
│   └── auto_rubric_prompt.txt                   # Auto-generated rubric
└── theorem_equiv_prompts/
    ├── iff_theorem_equiv_prompt_zero_shot.txt
    └── iff_theorem_equiv_prompt_one_shot.txt
```

## 🚀 Quick Start Example

### Method 1: Using the LLM Judge Function Directly

```python
#!/usr/bin/env python3
"""Score generated Lean code against golden reference using LLM judge"""

import sys
from pathlib import Path

# Setup paths for VeriBench
veribench_root = Path(__file__).parent / "self-opt-data-gen" / "veribench_bundle"
py_src = veribench_root / "veribench_dataset" / "py_src"
sys.path.insert(0, str(py_src))

from veribench.metrics.sim_file_eq_llm_judge import (
    get_sim_score_file_eq_llm_result_in_json_str
)

# Step 1: Prepare your files
agent_file = Path("outputs/my_model/counting_sort.lean")  # Your generated code
gold_file = Path("self-opt-data-gen/veribench_bundle/veribench_dataset/lean_src/veribench/cs_set/counting_sort.lean")

# Step 2: Load the prompt template
prompt_file = veribench_root / "veribench_prompts/eval_prompts/file_equiv_prompts/simplest_prompt_varying_score_range.txt"
prompt_template = prompt_file.read_text()

# Step 3: Get LLM judge score
result = get_sim_score_file_eq_llm_result_in_json_str(
    agent_filepath=agent_file,
    gold_filepath=gold_file,
    file_equiv_prompt_template=prompt_template,
    max_score=30,  # Score range: 0-30
    s_idx=0,       # Sample index (optional)
    n_idx=2,       # Task index (optional)
    relative_path=Path("cs_set/counting_sort.lean")
)

# Step 4: Display results
print(f"Score: {result['score']}/{result['max_score']}")
print(f"Normalized Score: {result['normalized_score']:.2f}")
print(f"\nRationale:\n{result['rationale']}")
```

### Method 2: Using the Multi-Metric Runner

```bash
# Set up environment variables
export TRACE_CUSTOMLLM_URL="https://your-llm-endpoint.com/v1"
export TRACE_CUSTOMLLM_API_KEY="your-api-key"
export TRACE_CUSTOMLLM_MODEL="anthropic.claude-3-5-sonnet-20240620-v1:0"

# Run evaluation with file equivalence metric
python self-opt-data-gen/veribench_bundle/tools/runner_multi_metrics.py \
    --dataset-root self-opt-data-gen/veribench_bundle/veribench_dataset \
    --output-root outputs/my_evaluation \
    --veribench-repo-root self-opt-data-gen/veribench_bundle \
    --metrics compile unit_tests file_equiv \
    --models-json models.json \
    --max-examples 10
```

## 📝 Prompt Templates Explained

### 1. Simple Prompt (0-10 scale)

**File:** `simplest_prompt.txt`

**Output Format:**
```json
<RESULTS>
{
  "rationale": "Explanation of matches/mismatches",
  "score": 8,
  "equivalent": false
}
</RESULTS>
```

**Use when:** You want a quick 0-10 score with JSON output.

### 2. Varying Score Range Prompt (0-N scale)

**File:** `simplest_prompt_varying_score_range.txt`

**Output Format:**
```xml
<RATIONALE>
The agent's definitions are semantically equivalent to the gold reference,
but a key theorem about associativity is missing.
</RATIONALE>

<SCORE_RESULT>
24
</SCORE_RESULT>
```

**Use when:** You want more granular scoring (e.g., 0-30) with cleaner format.

**Variables:**
- `{$GOLD_FILE}` - Golden reference Lean code
- `{$AGENT_FILE}` - Generated Lean code
- `{$MAX_SCORE}` - Maximum score (e.g., 30)

## 🔧 Complete Working Example

```python
#!/usr/bin/env python3
"""
Complete example: Evaluate generated Lean code for task 2 (counting_sort)
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Setup environment
os.environ["TRACE_CUSTOMLLM_URL"] = "https://your-endpoint.com/v1"
os.environ["TRACE_CUSTOMLLM_API_KEY"] = "your-api-key"
os.environ["TRACE_CUSTOMLLM_MODEL"] = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Add VeriBench to path
veribench_root = Path(__file__).parent / "self-opt-data-gen" / "veribench_bundle"
py_src = veribench_root / "veribench_dataset" / "py_src"
sys.path.insert(0, str(py_src))

from veribench.metrics.sim_file_eq_llm_judge import (
    get_sim_score_file_eq_llm_result_in_json_str
)


def evaluate_with_llm_judge(task_id: int, generated_lean_code: str, max_score: int = 30):
    """
    Evaluate generated Lean code using LLM judge.
    
    Args:
        task_id: Task ID (0-139)
        generated_lean_code: The generated Lean code (as string)
        max_score: Maximum score for evaluation (default: 30)
        
    Returns:
        Dictionary with score, rationale, and metadata
    """
    
    # Step 1: Get gold reference file path
    dataset_root = Path("veribench_dataset_utils/dataset")
    task_file = dataset_root / f"task_{task_id}.json"
    
    with open(task_file, 'r') as f:
        task_data = json.load(f)
    
    gold_lean_code = task_data['gold_reference_lean4_code']
    
    # Step 2: Save generated code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
        f.write(generated_lean_code)
        agent_file = Path(f.name)
    
    # Step 3: Save gold code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
        f.write(gold_lean_code)
        gold_file = Path(f.name)
    
    try:
        # Step 4: Load prompt template
        prompt_path = veribench_root / "veribench_prompts/eval_prompts/file_equiv_prompts/simplest_prompt_varying_score_range.txt"
        prompt_template = prompt_path.read_text()
        
        # Step 5: Get LLM judge score
        result = get_sim_score_file_eq_llm_result_in_json_str(
            agent_filepath=agent_file,
            gold_filepath=gold_file,
            file_equiv_prompt_template=prompt_template,
            max_score=max_score,
            s_idx=0,
            n_idx=task_id,
            relative_path=Path(f"task_{task_id}")
        )
        
        return result
        
    finally:
        # Cleanup temp files
        agent_file.unlink()
        gold_file.unlink()


def main():
    """Example usage"""
    
    # Example: Your generated Lean code
    generated_code = """
import Mathlib.Data.List.Sort

namespace CountingSort

def countingSort (l : List Nat) : List Nat :=
  if l.isEmpty then []
  else
    let max_val := l.foldl max 0
    let counts := List.range (max_val + 1) |>.map (fun i => l.count i)
    -- ... rest of implementation
    []

end CountingSort
"""
    
    print("="*80)
    print("LLM JUDGE EVALUATION")
    print("="*80)
    
    # Evaluate task 2 (counting_sort)
    result = evaluate_with_llm_judge(
        task_id=2,
        generated_lean_code=generated_code,
        max_score=30
    )
    
    # Display results
    print(f"\n📊 Score: {result['score']}/{result['max_score']}")
    print(f"📈 Normalized: {result['normalized_score']:.2%}")
    print(f"\n📝 Rationale:")
    print(result['rationale'])
    
    if result.get('error'):
        print(f"\n⚠️  Error: {result['error']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
```

## 🔑 Environment Variables Required

```bash
# Required for LLM judge
export TRACE_CUSTOMLLM_URL="https://your-llm-api-endpoint.com/v1"
export TRACE_CUSTOMLLM_API_KEY="your-api-key-here"
export TRACE_CUSTOMLLM_MODEL="anthropic.claude-3-5-sonnet-20240620-v1:0"

# Optional: For testing without calling LLM
export VERIBENCH_FAKE_LLM="1"  # Returns stub scores for dry-run testing
```

## 📊 Understanding the Results

### Result Dictionary Structure

```python
{
    'agent_file': str,          # Path to generated file
    'gold_file': str,           # Path to golden reference
    'score': int,               # Raw score (e.g., 24 out of 30)
    'normalized_score': float,  # Normalized to 0.0-1.0 (e.g., 0.8)
    'rationale': str,           # LLM's explanation
    'response': str,            # Full LLM response
    'error': str,               # Error message (if any)
    'max_score': int,           # Maximum possible score
    's_idx': int,               # Sample index
    'n_idx': int,               # Task index
    'relative_path': str        # Relative path for reference
}
```

### Score Interpretation

**For 0-10 scale:**
- `10`: Perfectly equivalent
- `8-9`: Very close, minor differences
- `6-7`: Similar intent, some differences
- `4-5`: Partial match
- `0-3`: Significantly different

**For 0-30 scale:**
- `30`: Perfectly equivalent
- `24-29`: Very close, minor differences
- `18-23`: Similar intent, some differences
- `12-17`: Partial match
- `0-11`: Significantly different

## 🎯 Integration with Your Pipeline

### Complete Evaluation Pipeline

```python
from guide.guide import VeribenchGuidewithUnitTests

def complete_evaluation(task_id: int, generated_lean_code: str):
    """Complete evaluation: compilation → unit tests → LLM judge"""
    
    # Step 1: Compilation & Unit Tests
    guide = VeribenchGuidewithUnitTests()
    python_code = load_python_code(task_id)
    
    score, feedback = guide.get_feedback(
        task=python_code,
        response=generated_lean_code,
        info=task_id
    )
    
    print(f"Compilation + Unit Tests: {score}")
    
    if score < 1.0:
        print(f"❌ Failed: {feedback}")
        return {
            'compilation_score': score,
            'llm_judge_score': None,
            'passed': False
        }
    
    # Step 2: LLM Judge (only if tests pass)
    llm_result = evaluate_with_llm_judge(task_id, generated_lean_code)
    
    return {
        'compilation_score': score,
        'llm_judge_score': llm_result['normalized_score'],
        'llm_rationale': llm_result['rationale'],
        'passed': True
    }
```

## 🛠️ Advanced Usage

### Custom Prompts

```python
# Create your own prompt template
custom_prompt = """
You are evaluating Lean 4 code for correctness and style.

<GOLD>
{$GOLD_FILE}
</GOLD>

<AGENT>
{$AGENT_FILE}
</AGENT>

Provide a score from 0 to {$MAX_SCORE}.

<RATIONALE>
Your explanation
</RATIONALE>

<SCORE_RESULT>
Score
</SCORE_RESULT>
"""

result = get_sim_score_file_eq_llm_result_in_json_str(
    agent_filepath=agent_file,
    gold_filepath=gold_file,
    file_equiv_prompt_template=custom_prompt,
    max_score=50  # Custom max score
)
```

### Batch Evaluation

```python
def batch_evaluate(task_ids: List[int], generated_codes: Dict[int, str]):
    """Evaluate multiple tasks in batch"""
    results = []
    
    for task_id in task_ids:
        if task_id not in generated_codes:
            continue
            
        result = evaluate_with_llm_judge(
            task_id=task_id,
            generated_lean_code=generated_codes[task_id]
        )
        
        results.append({
            'task_id': task_id,
            'score': result['normalized_score'],
            'rationale': result['rationale']
        })
    
    # Calculate average score
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"Average LLM Judge Score: {avg_score:.2%}")
    
    return results
```

## 📚 Related Documentation

- `guide/guide.py` - Compilation and unit test evaluation
- `VERIFIER_README.md` - Overall verification system
- `HOW_TO_FIND_GOLDEN_IMPLEMENTATIONS.md` - Finding golden references
- `veribench_bundle/README.md` - VeriBench bundle overview

## 💡 Tips

1. **Always evaluate compilation + unit tests first** - LLM judge is for semantic equivalence after basic correctness
2. **Use appropriate score ranges** - 0-10 for quick checks, 0-30 for more granular assessment
3. **Read the rationale** - The LLM's explanation often provides valuable insights
4. **Set environment variables** - Required for LLM API access
5. **Handle errors gracefully** - LLM calls can fail; check for errors in results
6. **Consider costs** - LLM judge requires API calls; use sparingly for large evaluations

## 🔍 Troubleshooting

### "No <RATIONALE> or <SCORE_RESULT> block found"
- The LLM didn't follow the expected format
- Try using a different prompt or adjusting temperature

### "Rate limit errors"
- The code includes automatic retry with exponential backoff
- Check your API rate limits

### "ModuleNotFoundError: veribench"
- Ensure you've added the py_src directory to sys.path
- Check that the path is correct

### Testing without LLM
```bash
# Use fake LLM for testing structure
export VERIBENCH_FAKE_LLM="1"
python your_script.py
```

