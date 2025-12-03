#!/usr/bin/env python3
"""
Verifier to verify the correctness of Lean 4 solutions generated for Veribench tasks.

This verifier:
1. Takes generated Lean 4 code from the agent
2. Uses PyPantograph to check if the code compiles and is valid
3. Validates the structure matches Veribench requirements
4. Provides detailed feedback on errors and correctness

Usage:
    python verifier.py --solution_file solution.lean
    python verifier.py --task_index 0 --solution "lean code here"
"""

import os
import re
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset


from pantograph.server import Server, ServerError, TacticFailure



class VeribenchVerifier:
    """Verifier for Veribench Lean 4 solutions."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the verifier.
        
        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose
        self.server = Server(imports=['Init', 'Std'])
        if self.verbose:
            print("✓ PyPantograph server initialized successfully")
    
    def load_veribench_task(self, task_index: int = 0) -> Dict[str, Any]:
        """Load a specific task from the Veribench dataset."""
        if self.verbose:
            print(f"Loading Veribench task {task_index}...")
        
        dataset = load_dataset("allenanie/veribench_with_prompts")
        task = dataset['train'][task_index]
        
        if self.verbose:
            print(f"✓ Loaded task: {task['filename']} (category: {task['category']})")
        
        return task
    
    def extract_lean_code(self, solution_text: str) -> Optional[str]:
        """Extract Lean 4 code from solution text (removes markdown formatting)."""
        # Look for ```lean code blocks
        lean_pattern = r'```lean\s*\n(.*?)\n```'
        matches = re.findall(lean_pattern, solution_text, re.DOTALL)
        
        if matches:
            # Return the first (and hopefully only) Lean code block
            return matches[0].strip()
        
        # If no markdown blocks found, assume the entire text is Lean code
        return solution_text.strip()
    
    
    def check_lean_syntax(self, lean_code: str) -> Dict[str, Any]:
        """
        Check if Lean code compiles successfully using PyPantograph.
        Pure compilation check: 1 if compiles, 0 if not.
        """
        compilation_result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'pantograph_available': True
        }
        
        try:
            # Try to compile the entire Lean code with PyPantograph
            # Create temporary file and attempt compilation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
                f.write(lean_code)
                temp_file = f.name
            
            try:
                # Attempt PyPantograph compilation
                # This is a simplified approach - just try to process the code
                compilation_result['is_valid'], error_messages = self._compile_with_pantograph(lean_code)
                
                if not compilation_result['is_valid']:
                    if error_messages:
                        compilation_result['errors'].extend(error_messages)
                    else:
                        compilation_result['errors'].append("Lean code failed to compile")
                    
            finally:
                # Clean up
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            compilation_result['errors'].append(f"Compilation failed: {str(e)}")
            compilation_result['is_valid'] = False
        
        return compilation_result
    
    def _compile_with_pantograph(self, lean_code: str) -> tuple[bool, list[str]]:
        """
        Compile Lean code using PyPantograph server.
        Returns (success, error_messages) tuple.
        """
        try:
            if not lean_code.strip():
                return False, ["Empty Lean code"]
            
            # Use PyPantograph's check_compile method
            try:
                compilation_units = self.server.check_compile(lean_code)
                
                error_messages = []
                # Check if there are any error messages in the compilation units
                for unit in compilation_units:
                    if hasattr(unit, 'messages') and unit.messages:
                        # Collect all error messages
                        for message in unit.messages:
                            if 'error:' in message.lower():
                                error_messages.append(message.strip())
                
                # If we found errors, return them
                if error_messages:
                    return False, error_messages
                
                # If we get here, no errors were found
                return True, []
                
            except (ServerError, TacticFailure) as e:
                # PyPantograph server error during compilation
                return False, [f"PyPantograph server error: {str(e)}"]
            except Exception as e:
                # Other compilation error
                return False, [f"Compilation error: {str(e)}"]
                
        except Exception as e:
            return False, [f"General error: {str(e)}"]
    
    def _extract_declarations(self, lean_code: str) -> List[str]:
        """Extract individual Lean declarations from code."""
        declarations = []
        current_decl = []
        
        lines = lean_code.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('--'):
                continue
            
            # Handle block comments
            if line.startswith('/-') or line.endswith('-/'):
                continue
            
            current_decl.append(line)
            
            # Check if this completes a declaration
            if (any(line.startswith(kw) for kw in ['def ', 'theorem ', 'lemma ', 'example ']) and
                (':=' in line or 'by' in line or line.endswith('sorry'))):
                declarations.append('\n'.join(current_decl))
                current_decl = []
            elif line.startswith('#'):
                # Evaluation commands
                declarations.append(line)
                current_decl = []
            elif line.startswith('namespace') or line.startswith('end'):
                # Namespace commands
                declarations.append(line)
                current_decl = []
        
        # Add any remaining declaration
        if current_decl:
            declarations.append('\n'.join(current_decl))
        
        return declarations
    
    
    def verify_solution(self, solution_text: str, task_index: Optional[int] = None, 
                       task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify a complete Lean 4 solution for a Veribench task.
        
        Args:
            solution_text: The generated Lean 4 solution
            task_index: Index of the task to verify against (optional)
            task: Task dictionary (optional, will load if not provided)
            
        Returns:
            Comprehensive verification results
        """
        if self.verbose:
            print("=" * 60)
            print("VERIBENCH SOLUTION VERIFICATION")
            print("=" * 60)
        
        # Load task if needed
        if task is None and task_index is not None:
            task = self.load_veribench_task(task_index)
        
        # Extract Lean code from solution
        lean_code = self.extract_lean_code(solution_text)
        if not lean_code:
            return {
                'success': False,
                'error': 'No Lean code found in solution',
                'lean_code': None
            }
        
        if self.verbose:
            print(f"✓ Extracted Lean code ({len(lean_code)} characters)")
        
        # Check compilation using PyPantograph
        compilation_result = self.check_lean_syntax(lean_code)
        
        # Compile overall results
        verification_result = {
            'success': compilation_result['is_valid'],
            'lean_code': lean_code,
            'compilation': compilation_result,
            'task_info': {
                'filename': task['filename'] if task else 'unknown',
                'category': task['category'] if task else 'unknown',
                'task_index': task_index
            },
            'overall_score': self._calculate_overall_score(compilation_result)
        }
        
        if self.verbose:
            self._print_verification_results(verification_result)
        
        return verification_result
    
    def _calculate_overall_score(self, compilation_result: Dict) -> float:
        """Calculate binary score based on compilation success (1.0 or 0.0)."""
        return 1.0 if compilation_result['is_valid'] else 0.0
    
    def _print_verification_results(self, result: Dict[str, Any]) -> None:
        """Print compilation-focused verification results."""
        print(f"\nTask: {result['task_info']['filename']} (Category: {result['task_info']['category']})")
        print(f"Overall Score: {result['overall_score']:.1f}/1.0")
        print(f"Success: {'✓' if result['success'] else '✗'}")
        
        print("\n🔍 LEAN COMPILATION:")
        compilation = result['compilation']
        print(f"  Compilation successful: {'✓' if compilation['is_valid'] else '✗'}")
        print(f"  PyPantograph available: {'✓' if compilation['pantograph_available'] else '✗'}")
        
        if compilation['errors']:
            print("  Compilation Errors:")
            for error in compilation['errors']:
                print(f"    - {error}")
        
        if compilation['warnings']:
            print("  Compilation Warnings:")
            for warning in compilation['warnings']:
                print(f"    - {warning}")
        
        print("\n" + "=" * 60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Verify Veribench Lean 4 solutions")
    parser.add_argument('--task_index', type=int, default=0, 
                       help='Index of the Veribench task to verify against')
    parser.add_argument('--solution_file', type=str,
                       help='Path to file containing the Lean 4 solution')
    parser.add_argument('--solution', type=str,
                       help='Lean 4 solution as a string')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--json_output', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Get solution text
    if args.solution_file:
        with open(args.solution_file, 'r') as f:
            solution_text = f.read()
    elif args.solution:
        solution_text = args.solution
    else:
        print("Error: Must provide either --solution_file or --solution")
        sys.exit(1)
    
    # Create verifier and run verification
    verifier = VeribenchVerifier(verbose=not args.quiet)
    result = verifier.verify_solution(solution_text, task_index=args.task_index)
    
    if args.json_output:
        # Remove non-serializable elements for JSON output
        json_result = {k: v for k, v in result.items() if k != 'lean_code'}
        print(json.dumps(json_result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

from lean_interpretor import lean_interpreter

class LeanGuide(AutoGuide):
    """
    Custom guide that uses the eval_metric function to evaluate responses
    and provide feedback for the BigBench tasks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, task, response, info, **kwargs):
        try:
            result = lean_interpreter(response)
            correctness = result["valid"]
            score = 1.0 if correctness else 0.0

            if correctness:
                feedback = "The answer is correct! No need to change anything."
            else:
                error_message = "\n\n".join(result["error_message"])
                summary_message = result["summary"]
                feedback = f'The answer is wrong. We expect the output of your answer to be "{info}". Please modify the prompt and relevant parts of the program to help LLM produce the right answer.'

            return score, feedback

        except Exception as e:
            return 0.0, f"Error occurred: {str(e)}. Please fix the error and try again."

    def metric(self, task, response, info, **kwargs):
        score, _ = self.forward(task, response, info, **kwargs)
        return score

if __name__ == "__main__":
    main()
