#!/usr/bin/env python3
"""
Add test scores to OpenEvolve history by evaluating all lean programs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import hashlib

# Setup paths
_script_dir = Path(__file__).resolve().parent
_veribench_dataset_utils = _script_dir / "veribench_dataset_utils"

if str(_veribench_dataset_utils) not in sys.path:
    sys.path.insert(0, str(_veribench_dataset_utils))

from eval_utils import evaluate

def hash_lean_code(code: str) -> str:
    """Create a hash of the lean code for caching."""
    return hashlib.md5(code.encode('utf-8')).hexdigest()

def evaluate_and_add_scores(
    run_num: int,
    task_id: int,
    result_file: Path,
    cache: Dict[Tuple[int, str], Tuple[float, str]]
) -> Tuple[int, int]:
    """
    Evaluate all lean programs in a result file and add test scores.
    
    Args:
        run_num: OpenEvolve run number (1-3)
        task_id: Task ID (10-50)
        result_file: Path to the JSON result file
        cache: Cache dictionary mapping (task_id, code_hash) to (score, feedback)
        
    Returns:
        Tuple of (num_evaluated, num_cached)
    """
    if not result_file.exists():
        print(f"  ⚠️  File not found: {result_file}")
        return 0, 0
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ Failed to parse JSON: {result_file}")
        print(f"     Error: {e}")
        return 0, 0
    
    history = data.get('history', [])
    if not history:
        print(f"  ⚠️  No history found")
        return 0, 0
    
    print(f"  📊 Processing {len(history)} attempts")
    
    num_evaluated = 0
    num_cached = 0
    
    for attempt_entry in history:
        lean_code = attempt_entry.get('best_lean_program', '')
        attempt_num = attempt_entry.get('attempt', '?')
        
        # Check cache using code hash (evaluate ALL code, even empty/placeholder)
        code_hash = hash_lean_code(lean_code)
        cache_key = (task_id, code_hash)
        
        if cache_key in cache:
            score, feedback = cache[cache_key]
            num_cached += 1
        else:
            # Evaluate ALL code
            print(f"    Evaluating attempt {attempt_num}...", end='', flush=True)
            try:
                score, feedback = evaluate(task_id, lean_code)
                cache[cache_key] = (score, feedback)
                num_evaluated += 1
                print(f" Score: {score:.3f}")
            except Exception as e:
                print(f" ❌ Error: {e}")
                score = None
                feedback = f"Evaluation error: {str(e)}"
        
        # Add test_score to attempt
        attempt_entry['test_score'] = score
    
    # Write back to file
    try:
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ Saved updated file")
    except Exception as e:
        print(f"  ❌ Failed to save file: {e}")
        return 0, 0
    
    return num_evaluated, num_cached

def main():
    """Main processing function."""
    print("=" * 80)
    print("ADDING TEST SCORES TO OPENEVOLVE HISTORY")
    print("=" * 80)
    
    base_dir = Path(__file__).resolve().parent / "results_llm_judge"
    
    # Cache for evaluation results
    # Key: (task_id, code_hash), Value: (score, feedback)
    cache: Dict[Tuple[int, str], Tuple[float, str]] = {}
    
    total_evaluated = 0
    total_cached = 0
    total_files = 0
    
    # Process runs 1-3, tasks 10-50
    for run_num in range(1, 4):
        print(f"\n{'='*80}")
        print(f"Processing OpenEvolve Run {run_num}")
        print(f"{'='*80}")
        
        run_dir = base_dir / f"openevolve_{run_num}"
        
        if not run_dir.exists():
            print(f"  ⚠️  Directory not found: {run_dir}")
            continue
        
        for task_id in range(10, 51):
            result_file = run_dir / f"openevolve_task_{task_id}_result.json"
            print(f"\n📁 openevolve_{run_num}/openevolve_task_{task_id}_result.json")
            
            total_files += 1
            num_evaluated, num_cached = evaluate_and_add_scores(
                run_num, task_id, result_file, cache
            )
            
            total_evaluated += num_evaluated
            total_cached += num_cached
            
            # Print cache stats periodically
            if task_id % 5 == 0:
                print(f"  💾 Cache size: {len(cache)} unique programs")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Total evaluations performed: {total_evaluated}")
    print(f"Total cache hits: {total_cached}")
    if (total_evaluated + total_cached) > 0:
        print(f"Cache efficiency: {total_cached / (total_evaluated + total_cached) * 100:.1f}%")
    print(f"Unique programs in cache: {len(cache)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
