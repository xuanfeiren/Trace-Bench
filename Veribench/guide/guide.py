import requests
from opto.trainer.guide import Guide
from my_processing_agents.lean_interpretor import lean_interpreter
from opto.optimizers.utils import print_color

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

class VeribenchGuide(Guide):
    """
    Guide that uses lean_interpreter to evaluate Veribench responses
    and provide feedback.
    """

    def __init__(self):
        super().__init__()

    def get_feedback(self, task, response, info=None, **kwargs):
        """
        Get feedback from the agent's Lean code response.
        
        Args:
            task: The task being evaluated (user query)
            response: The LLM-generated Lean code response
            info: Additional info (optional)
            
        Returns:
            Tuple of (score, feedback)
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
            error_str = str(e)
            # Check for system errors that should be raised, not returned as feedback
            system_errors = [
                "event loop is already running",
                "event loop",
                "asyncio",
                "RuntimeError",
            ]
            if any(err.lower() in error_str.lower() for err in system_errors):
                print_color(f"System error (not a Lean compilation error): {error_str}. This is likely due to asyncio conflicts.", "red")
                raise RuntimeError(
                    f"System error (not a Lean compilation error): {error_str}. "
                    "This is likely due to asyncio conflicts."
                ) from e
            return 0.0, f"Error occurred: {error_str}. Please fix the error and try again."

    def metric(self, task, response, info=None, **kwargs):
        """Metric for the agent's performance."""
        score, _ = self.get_feedback(task, response, info, **kwargs)
        return score


class WebGuide(Guide):
    """
    Guide that uses a web server to evaluate Veribench responses.
    Calls the Lean Feedback Server via HTTP to get compilation feedback.
    
    This avoids asyncio conflicts by delegating Lean compilation to a separate process.
    
    Usage:
        # First start the server:
        # uv run python webserver/lean_feedback_server.py --port 8000
        
        guide = WebGuide(server_url="http://localhost:8000")
        score, feedback = guide.get_feedback(task, response)
    """

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 20):
        """
        Initialize WebGuide.
        
        Args:
            server_url: URL of the Lean Feedback Server
            timeout: Request timeout in seconds (default 70, Lean Server has 60s internal timeout)
        """
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
    
    def _check_server(self) -> bool:
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_feedback(self, task, response, info=None, **kwargs):
        """
        Get feedback from the agent's Lean code response via web server.
        
        Args:
            task: The task being evaluated (user query)
            response: The LLM-generated Lean code response
            info: Additional info (optional)
            
        Returns:
            Tuple of (score, feedback)
        """
        try:
            # Call the web server
            http_response = requests.post(
                f"{self.server_url}/feedback",
                json={
                    "lean_code": response,
                    "remove_import_errors": True
                },
                timeout=self.timeout
            )
            
            if http_response.status_code != 200:
                error_detail = http_response.json().get("detail", "Unknown error")
                return 0.0, f"Server error ({http_response.status_code}): {error_detail}"
            
            data = http_response.json()
            score = data["score"]
            feedback = data["feedback"]
            
            print_color(feedback, "yellow")
            return score, feedback

        except requests.exceptions.ConnectionError:
            error_msg = (
                f"Cannot connect to Lean Feedback Server at {self.server_url}. "
                "Please start the server first:\n"
                "  uv run python webserver/lean_feedback_server.py --port 8000"
            )
            print_color(error_msg, "red")
            return 0.0, error_msg
        
        except requests.exceptions.Timeout:
            error_msg = f"""Compilation TIMEOUT after {self.timeout} seconds.

        This usually means the Lean code contains constructs that are extremely slow to compile:

        COMMON CAUSES:
        1. 'native_decide' tactic - This executes code at compile time and is VERY slow for recursive functions
        2. 'decide' on large data - Similar issue with compile-time evaluation
        3. Complex recursive functions without 'termination_by' - Lean struggles to prove termination

        HOW TO FIX:
        1. Replace 'native_decide' with 'sorry' or 'rfl' where applicable
        2. Use '#eval!' instead of 'example ... := by native_decide' for testing
        3. Add 'termination_by <measure>' and 'decreasing_by all_goals sorry' for recursive functions
        4. Simplify proofs - use 'sorry' as placeholder for complex theorems

        EXAMPLE FIX:
        -- BAD (causes timeout):
        example : myFunc #[1,2,3] = some 1 := by native_decide
        
        -- GOOD (fast):
        #eval! myFunc #[1,2,3]  -- just evaluate, don't prove
        -- OR
        example : myFunc #[1,2,3] = some 1 := sorry  -- placeholder proof
        """
            print_color(f"TIMEOUT: {self.timeout}s exceeded", "red")
            print_color(f'Lean code that caused timeout: {response}', "red")
            return 0.0, error_msg
            
        except Exception as e:
            error_str = str(e)
            print_color(f"WebGuide error: {error_str}", "red")
            return 0.0, f"Error occurred: {error_str}. Please fix the error and try again."

    def metric(self, task, response, info=None, **kwargs):
        """Metric for the agent's performance."""
        score, _ = self.get_feedback(task, response, info, **kwargs)
        return score
