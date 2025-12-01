#!/usr/bin/env python3
"""
Integration script that combines the Veribench agent with the verifier.
Generates solutions and immediately verifies them.
"""

import os
import sys
import json
import argparse
from agent import solve_veribench_task
from verifier import VeribenchVerifier
from datasets import load_dataset
import litellm

def generate_and_verify_solution(task_index: int = 0, save_results: bool = True) -> dict:
    """
    Generate a solution using the agent and verify it.
    
    Args:
        task_index: Index of the Veribench task to solve
        save_results: Whether to save results to files
        
    Returns:
        Dictionary containing generation and verification results
    """
    print("=" * 80)
    print(f"VERIBENCH AGENT + VERIFIER PIPELINE")
    print("=" * 80)
    
    # Load the task
    dataset = load_dataset("allenanie/veribench_with_prompts")
    task = dataset['train'][task_index]
    
    print(f"Task {task_index}: {task['filename']} (Category: {task['category']})")
    print(f"Python code length: {len(task['code'])} characters")
    
    # Generate solution using the agent
    print("\n🤖 GENERATING SOLUTION...")
    print("-" * 40)
    
    try:
        # Build prompts
        system_prompt = task.get("system_prompt_with_examples", task.get("system_prompt", ""))
        user_query = task.get("user_query", "")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Call LLM
        response = litellm.completion(
            model="gemini/gemini-2.5-flash-lite",
            messages=messages,
            max_tokens=65536,
            temperature=0.1
        )
        
        generated_solution = response.choices[0].message.content
        
        print(f"✓ Solution generated ({len(generated_solution)} characters)")
        
        # Save generated solution if requested
        if save_results:
            solution_file = f"solution_task_{task_index}.md"
            with open(solution_file, 'w') as f:
                f.write(f"# Veribench Task {task_index}: {task['filename']}\n\n")
                f.write(f"**Category:** {task['category']}\n\n")
                f.write(f"**Generated Solution:**\n\n")
                f.write(generated_solution)
            print(f"✓ Solution saved to: {solution_file}")
        
    except Exception as e:
        print(f"✗ Error generating solution: {e}")
        return {
            'success': False,
            'error': f"Generation failed: {e}",
            'task_index': task_index,
            'task_info': {
                'filename': task['filename'],
                'category': task['category']
            }
        }
    
    # Verify the solution
    print("\n🔍 VERIFYING SOLUTION...")
    print("-" * 40)
    
    verifier = VeribenchVerifier(verbose=True)
    verification_result = verifier.verify_solution(generated_solution, task_index=task_index, task=task)
    
    # Compile complete results
    complete_result = {
        'task_index': task_index,
        'task_info': {
            'filename': task['filename'],
            'category': task['category'],
            'code_length': len(task['code'])
        },
        'generation': {
            'success': True,
            'solution_length': len(generated_solution),
            'model': "gemini/gemini-2.5-flash-lite"
        },
        'verification': verification_result,
        'overall_success': verification_result['success'],
        'pipeline_score': verification_result['overall_score']
    }
    
    # Save complete results if requested
    if save_results:
        results_file = f"results_task_{task_index}.json"
        with open(results_file, 'w') as f:
            # Remove the actual lean code for JSON serialization
            json_result = {k: v for k, v in complete_result.items()}
            if 'lean_code' in json_result['verification']:
                del json_result['verification']['lean_code']
            json.dump(json_result, f, indent=2)
        print(f"✓ Results saved to: {results_file}")
    
    # Print summary
    print("\n📊 PIPELINE SUMMARY")
    print("-" * 40)
    print(f"Task: {task['filename']}")
    print(f"Generation: {'✓ Success' if complete_result['generation']['success'] else '✗ Failed'}")
    print(f"Verification: {'✓ Success' if complete_result['verification']['success'] else '✗ Failed'}")
    print(f"Overall Score: {complete_result['pipeline_score']:.2f}/1.00")
    print(f"Compilation Score: {verification_result['overall_score']:.1f}/1.0")
    
    if verification_result['compilation']['errors']:
        print("Compilation errors found:")
        for error in verification_result['compilation']['errors']:
            print(f"  - {error}")
    
    if verification_result['compilation']['warnings']:
        print("Compilation warnings:")
        for warning in verification_result['compilation']['warnings']:
            print(f"  - {warning}")
    
    return complete_result

def run_batch_evaluation(start_index: int = 0, num_tasks: int = 5) -> dict:
    """
    Run the agent+verifier pipeline on multiple tasks.
    
    Args:
        start_index: Starting task index
        num_tasks: Number of tasks to evaluate
        
    Returns:
        Batch evaluation results
    """
    print("=" * 80)
    print(f"BATCH EVALUATION: Tasks {start_index} to {start_index + num_tasks - 1}")
    print("=" * 80)
    
    batch_results = {
        'start_index': start_index,
        'num_tasks': num_tasks,
        'task_results': [],
        'summary': {
            'total_tasks': 0,
            'successful_generations': 0,
            'successful_verifications': 0,
            'average_score': 0.0,
            'average_compilation_score': 0.0
        }
    }
    
    total_score = 0.0
    total_compilation_score = 0.0
    
    for i in range(num_tasks):
        task_index = start_index + i
        print(f"\n{'='*20} Task {task_index} {'='*20}")
        
        try:
            result = generate_and_verify_solution(task_index, save_results=False)
            batch_results['task_results'].append(result)
            
            if result.get('generation', {}).get('success', False):
                batch_results['summary']['successful_generations'] += 1
            
            if result.get('verification', {}).get('success', False):
                batch_results['summary']['successful_verifications'] += 1
            
            total_score += result.get('pipeline_score', 0.0)
            total_compilation_score += result.get('verification', {}).get('overall_score', 0.0)
            
        except Exception as e:
            print(f"✗ Error processing task {task_index}: {e}")
            batch_results['task_results'].append({
                'task_index': task_index,
                'error': str(e),
                'overall_success': False,
                'pipeline_score': 0.0
            })
    
    # Calculate summary statistics
    batch_results['summary']['total_tasks'] = num_tasks
    batch_results['summary']['average_score'] = total_score / num_tasks
    batch_results['summary']['average_compilation_score'] = total_compilation_score / num_tasks
    
    # Print batch summary
    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)
    summary = batch_results['summary']
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Successful generations: {summary['successful_generations']}/{summary['total_tasks']} ({summary['successful_generations']/summary['total_tasks']*100:.1f}%)")
    print(f"Successful verifications: {summary['successful_verifications']}/{summary['total_tasks']} ({summary['successful_verifications']/summary['total_tasks']*100:.1f}%)")
    print(f"Average pipeline score: {summary['average_score']:.3f}/1.000")
    print(f"Average compilation score: {summary['average_compilation_score']:.3f}/1.000")
    
    # Save batch results
    batch_file = f"batch_results_{start_index}_{start_index + num_tasks - 1}.json"
    with open(batch_file, 'w') as f:
        # Remove lean_code for JSON serialization
        json_results = batch_results.copy()
        for task_result in json_results['task_results']:
            if 'verification' in task_result and 'lean_code' in task_result['verification']:
                del task_result['verification']['lean_code']
        json.dump(json_results, f, indent=2)
    print(f"✓ Batch results saved to: {batch_file}")
    
    return batch_results

def run_first_10_tasks() -> dict:
    """
    Run the agent+verifier pipeline on the first 10 tasks and calculate mean score.
    
    Returns:
        Results with mean score calculation
    """
    print("=" * 80)
    print("RUNNING FIRST 10 VERIBENCH TASKS")
    print("=" * 80)
    
    results = {
        'total_tasks': 10,
        'task_results': [],
        'binary_scores': [],
        'successful_generations': 0,
        'successful_compilations': 0,
        'mean_score': 0.0
    }
    
    total_score = 0.0
    
    for task_index in range(10):
        print(f"\n{'='*20} Task {task_index} {'='*20}")
        
        try:
            result = generate_and_verify_solution(task_index, save_results=False)
            results['task_results'].append(result)
            
            # Extract binary score (1.0 or 0.0)
            binary_score = result.get('pipeline_score', 0.0)
            results['binary_scores'].append(binary_score)
            total_score += binary_score
            
            if result.get('generation', {}).get('success', False):
                results['successful_generations'] += 1
            
            if result.get('verification', {}).get('success', False):
                results['successful_compilations'] += 1
            
            print(f"Task {task_index} Score: {binary_score}")
            
        except Exception as e:
            print(f"✗ Error processing task {task_index}: {e}")
            results['task_results'].append({
                'task_index': task_index,
                'error': str(e),
                'overall_success': False,
                'pipeline_score': 0.0
            })
            results['binary_scores'].append(0.0)
    
    # Calculate mean score
    results['mean_score'] = total_score / 10.0
    
    # Print summary
    print("\n" + "=" * 80)
    print("FIRST 10 TASKS SUMMARY")
    print("=" * 80)
    print(f"Total tasks processed: {results['total_tasks']}")
    print(f"Successful generations: {results['successful_generations']}/10 ({results['successful_generations']/10*100:.1f}%)")
    print(f"Successful compilations: {results['successful_compilations']}/10 ({results['successful_compilations']/10*100:.1f}%)")
    print(f"Binary scores: {results['binary_scores']}")
    print(f"Mean score: {results['mean_score']:.3f}/1.000")
    
    # Save results
    results_file = "first_10_tasks_results.json"
    with open(results_file, 'w') as f:
        # Remove lean_code for JSON serialization
        json_results = results.copy()
        for task_result in json_results['task_results']:
            if 'verification' in task_result and 'lean_code' in task_result['verification']:
                del task_result['verification']['lean_code']
        json.dump(json_results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")
    
    return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Veribench Agent + Verifier Pipeline")
    parser.add_argument('--task_index', type=int, default=None,
                       help='Index of the Veribench task to solve (single task mode)')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch evaluation on multiple tasks')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Starting task index for batch evaluation')
    parser.add_argument('--num_tasks', type=int, default=5,
                       help='Number of tasks for batch evaluation')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--first_10', action='store_true',
                       help='Run first 10 tasks and calculate mean score')
    
    args = parser.parse_args()
    
    if args.first_10:
        run_first_10_tasks()
    elif args.batch:
        run_batch_evaluation(args.start_index, args.num_tasks)
    elif args.task_index is not None:
        generate_and_verify_solution(args.task_index, save_results=not args.no_save)
    else:
        # Default behavior: run first 10 tasks
        print("No specific mode selected. Running first 10 tasks by default...")
        run_first_10_tasks()

if __name__ == "__main__":
    main()
