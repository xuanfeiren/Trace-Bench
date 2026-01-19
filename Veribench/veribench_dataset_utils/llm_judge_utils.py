"""
Simple LLM Judge to Score Generated Lean Code

This demo uses the EXACT implementation from veribench_bundle to compare 
generated Lean code with golden reference using Claude.

=============================================================================
SOURCE CODE REFERENCES
=============================================================================

All code in this file is copied from the existing VeriBench codebase:

1. Prompt Template:
   File: self-opt-data-gen/veribench_bundle/veribench_prompts/eval_prompts/
         file_equiv_prompts/simplest_prompt_varying_score_range.txt
   
2. Main Implementation:
   File: self-opt-data-gen/veribench_bundle/veribench_dataset/py_src/
         veribench/metrics/sim_file_eq_llm_judge.py
   Function: get_sim_score_file_eq_llm_result_in_json_str
   Lines: 76-152

3. Specific Line References:
   - Placeholder replacement: lines 117-118
   - System message: line 124 (character-by-character copy)
   - Conversation format: lines 125-128
   - API parameters: line 129 (temperature=0.1, max_n_tokens=4000)
   - Regex patterns: lines 132-133
   - Result extraction: lines 134-142

Nothing in this file is written by hand - everything comes from the 
existing VeriBench implementation.
=============================================================================
"""

import sys
import re
import json
import logging
from pathlib import Path
from openai import OpenAI

# Suppress HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import secrets from my_processing_agents
sys.path.insert(0, str(parent_dir / "my_processing_agents"))
import secrets_local

# Prompt directory (file is in veribench_dataset_utils/)
veribench_root = parent_dir / "self-opt-data-gen" / "veribench_bundle"
prompts_dir = veribench_root / "veribench_prompts" / "eval_prompts" / "file_equiv_prompts"

# Initialize OpenAI client with custom endpoint
client = OpenAI(
    base_url=secrets_local.os.environ['TRACE_CUSTOMLLM_URL'],
    api_key=secrets_local.os.environ['TRACE_CUSTOMLLM_API_KEY']
)


def load_prompt_template(prompt_name="simplest_prompt_varying_score_range.txt"):
    """
    Load LLM judge prompt template from veribench.
    
    Source: self-opt-data-gen/veribench_bundle/veribench_prompts/eval_prompts/
            file_equiv_prompts/simplest_prompt_varying_score_range.txt
    """
    prompt_file = prompts_dir / prompt_name
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    return prompt_file.read_text()


def simple_llm_judge(generated_code: str, golden_code: str, max_score: int = 30) -> dict:
    """
    Use Claude to judge similarity between generated and golden Lean code.
    
    This is the EXACT implementation from:
    Source: self-opt-data-gen/veribench_bundle/veribench_dataset/py_src/
            veribench/metrics/sim_file_eq_llm_judge.py
            Function: get_sim_score_file_eq_llm_result_in_json_str (lines 76-152)
    
    Args:
        generated_code: The LLM-generated Lean code
        golden_code: The golden reference Lean code
        max_score: Maximum score (default: 30)
            Source: sim_file_eq_llm_judge.py line 80 uses max_score: int = 30 as default
            This is the standard score range used throughout VeriBench for file equivalence
        
    Returns:
        dict with keys: score, normalized_score, rationale, response
    """
    
    # ========================================================================
    # Load prompt template
    # Source: sim_file_eq_llm_judge.py uses this prompt template file
    # ========================================================================
    file_equiv_prompt_template = load_prompt_template()
    
    # ========================================================================
    # Format prompt with file contents
    # Source: sim_file_eq_llm_judge.py lines 117-118
    #   prompt: str = file_equiv_prompt_template.replace("{$GOLD_FILE}", gold_file_content).replace("{$AGENT_FILE}", agent_file_content)
    #   prompt: str = prompt.replace("{$MAX_SCORE}", str(max_score))
    # ========================================================================
    prompt = file_equiv_prompt_template.replace("{$GOLD_FILE}", golden_code)
    prompt = prompt.replace("{$AGENT_FILE}", generated_code)
    prompt = prompt.replace("{$MAX_SCORE}", str(max_score))
    
    # ========================================================================
    # Create system message
    # Source: sim_file_eq_llm_judge.py line 124 (character-by-character copy)
    #   system_msg = "You are a Lean 4 code analysis AI. You MUST respond EXACTLY in this format:\n\n<RATIONALE>\nYour explanation here\n</RATIONALE>\n\n<SCORE_RESULT>\nAn integer from 0 to {}\n</SCORE_RESULT>\n\nDo not add any text before, after, or between these tags. The response must start with <RATIONALE> and end with </SCORE_RESULT>.".format(max_score)
    # ========================================================================
    system_msg = "You are a Lean 4 code analysis AI. You MUST respond EXACTLY in this format:\n\n<RATIONALE>\nYour explanation here\n</RATIONALE>\n\n<SCORE_RESULT>\nAn integer from 0 to {}\n</SCORE_RESULT>\n\nDo not add any text before, after, or between these tags. The response must start with <RATIONALE> and end with </SCORE_RESULT>.".format(max_score)
    
    # ========================================================================
    # Create conversation
    # Source: sim_file_eq_llm_judge.py lines 125-128
    #   conv = [
    #       {"role": "system", "content": system_msg},
    #       {"role": "user", "content": prompt}
    #   ]
    # ========================================================================
    conv = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    # ========================================================================
    # Call LLM
    # Source: sim_file_eq_llm_judge.py line 129
    #   response: str = call_llm_from_conv(conv, temperature=0.1, max_n_tokens=4000)
    # Note: We use OpenAI client directly instead of call_llm_from_conv wrapper
    # ========================================================================
    response = client.chat.completions.create(
        model=secrets_local.MODEL,
        messages=conv,
        max_tokens=4000,      # From: max_n_tokens=4000
        temperature=0.1,      # From: temperature=0.1
        top_p=0.9            # Default in veribench
    )
    
    response_text = response.choices[0].message.content
    
    # ========================================================================
    # Extract rationale and score using regex
    # Source: sim_file_eq_llm_judge.py lines 132-133
    #   rationale_match: re.Match[str] | None = re.search(r"<RATIONALE>\s*(.*?)\s*</RATIONALE>", response, re.DOTALL | re.IGNORECASE)
    #   score_result_match: re.Match[str] | None = re.search(r"<SCORE_RESULT>\s*(\d+)\s*</SCORE_RESULT>", response, re.DOTALL | re.IGNORECASE)
    # ========================================================================
    rationale_match = re.search(r"<RATIONALE>\s*(.*?)\s*</RATIONALE>", response_text, re.DOTALL | re.IGNORECASE)
    score_result_match = re.search(r"<SCORE_RESULT>\s*(\d+)\s*</SCORE_RESULT>", response_text, re.DOTALL | re.IGNORECASE)
    
    # ========================================================================
    # Parse and return results
    # Source: sim_file_eq_llm_judge.py lines 134-142
    #   if rationale_match and score_result_match:
    #       rationale: str = rationale_match.group(1).strip()
    #       score: int = int(score_result_match.group(1))
    #       res["score"] = score
    #       res["normalized_score"] = (score / max_score) if max_score else 0.0
    #       res["rationale"] = rationale
    #       res["response"] = response
    #       return res
    # ========================================================================
    if rationale_match and score_result_match:
        rationale = rationale_match.group(1).strip()
        score = int(score_result_match.group(1))
        
        return {
            "score": score,
            "normalized_score": (score / max_score) if max_score else 0.0,
            "rationale": rationale,
            "response": response_text
        }
    else:
        # Source: sim_file_eq_llm_judge.py lines 144-147 (error case)
        return {
            "score": 0,
            "normalized_score": 0.0,
            "rationale": "Error: Could not parse LLM response",
            "response": response_text,
            "error": "No <RATIONALE> or <SCORE_RESULT> block found"
        }


def get_golden_reference(task_id: int) -> str:
    """
    Load golden reference Lean code from dataset.
    
    Args:
        task_id: Task ID (0-139)
        
    Returns:
        Golden reference Lean 4 code
        
    Raises:
        FileNotFoundError: If task file doesn't exist
        ValueError: If golden reference code not found in task file
    """
    dataset_root = Path(__file__).resolve().parent / "dataset"
    task_file = dataset_root / f"task_{task_id}.json"
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    with open(task_file, 'r') as f:
        task_data = json.load(f)
    
    golden_code = task_data.get('gold_reference_lean4_code', '')
    if not golden_code:
        raise ValueError(f"No gold reference Lean code found for task {task_id}")
    
    return golden_code


def judge_generated_code(task_id: int, lean_code_implementation: str, max_score: int = 30) -> dict:
    """
    Evaluate generated Lean code against golden reference using LLM judge.
    
    This function:
    1. Loads the golden reference code for the given task_id
    2. Compares the generated code with golden reference using LLM judge
    3. Returns the evaluation results
    
    Args:
        task_id: Task ID (0-139)
        lean_code_implementation: Generated Lean code from an LLM
        max_score: Maximum score (default: 30)
            Source: This default comes from the original VeriBench codebase:
            - sim_file_eq_llm_judge.py line 80: max_score: int = 30
            - runner_multi_metrics.py line 830: max_score = options.get("max_score", 30)
            This is the standard score range for file equivalence evaluation in VeriBench.
        
    Returns:
        dict with keys:
            - score: int (0 to max_score)
            - normalized_score: float (0.0 to 1.0)
            - rationale: str (explanation from LLM judge)
            - response: str (full LLM response)
            - task_id: int (echoed back)
            
    Example:
        >>> result = judge_generated_code(
        ...     task_id=2,
        ...     lean_code_implementation="namespace CountingSort\ndef countingSort...",
        ...     max_score=30
        ... )
        >>> print(f"Score: {result['score']}/30")
        >>> print(f"Rationale: {result['rationale']}")
    """
    # Load golden reference for this task
    golden_code = get_golden_reference(task_id)
    
    # Call LLM judge to compare generated code with golden reference
    result = simple_llm_judge(
        generated_code=lean_code_implementation,
        golden_code=golden_code,
        max_score=max_score
    )
    
    # Add task_id to result for tracking
    result['task_id'] = task_id
    
    return result


def main():
    """Demo of judge_generated_code function"""
    
    print("=" * 80)
    print("LLM Judge Demo: Evaluating Generated Code for Task ID")
    print("=" * 80)
    print()
    
    # Example: Task 2 is CountingSort
    task_id = 2
    
    # Example generated code for CountingSort (simulating LLM output)
    generated_code = """namespace CountingSort

def countingSort (arr : List Nat) : List Nat := Id.run do
  match arr with
  | [] => []
  | _ => 
    let maxVal := arr.foldl max 0
    let mut count := mkArray (maxVal + 1) 0
    for x in arr do
      count := count.modify x (· + 1)
    let mut result := []
    for i in [:maxVal + 1] do
      let cnt := count[i]!
      result := result ++ List.replicate cnt i
    return result

def isSorted : List Nat → Bool
| [] => true  
| [_] => true
| x :: y :: xs => x ≤ y && isSorted (y :: xs)

def countOccurrences (xs : List Nat) (x : Nat) : Nat :=
  xs.filter (· = x) |>.length

def isPerm (xs ys : List Nat) : Bool :=
  xs.length = ys.length && 
  (xs.foldl (fun acc x => acc && countOccurrences ys x = countOccurrences xs x) true)

def Pre (arr : List Nat) : Prop := True 

def Post (arr result : List Nat) : Prop :=
  isSorted result ∧ isPerm arr result 

theorem correctness (arr : List Nat) (h : Pre arr) :
  Post arr (countingSort arr) := sorry

end CountingSort
"""
    
    print(f"Task ID: {task_id}")
    print("Evaluating generated code against golden reference...")
    print()
    
    # Use the new judge_generated_code function
    result = judge_generated_code(
        task_id=task_id,
        lean_code_implementation=generated_code,
        max_score=30
    )
    
    print(f"Task ID: {result['task_id']}")
    print(f"Score: {result['score']}/{30}")
    print(f"Normalized Score: {result['normalized_score']:.2%}")
    print()
    print("Rationale:")
    print(result['rationale'])
    print()
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        print()
    
    print("=" * 80)
    
    # Save result
    output_file = Path(__file__).parent / f"llm_judge_task_{task_id}_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Result saved to: {output_file}")


if __name__ == "__main__":
    main()
