"""
Lean Feedback Web Server

A FastAPI web server that evaluates Lean 4 code and returns feedback.
Provides an HTTP endpoint to check Lean code compilation and get structured feedback.

OPTIMIZED VERSION: 
- Uses a global Lean Server instance for faster responses
- Auto-restarts the server if it crashes

Usage:
    python lean_feedback_server.py [--host HOST] [--port PORT]
    
Example:
    python lean_feedback_server.py --port 8000
    
    # Then send a POST request:
    curl -X POST "http://localhost:8000/feedback" \
         -H "Content-Type: application/json" \
         -d '{"lean_code": "def hello := 42"}'
"""

import sys
import os
import argparse
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add my_processing_agents to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'my_processing_agents'))

from pantograph.server import Server
from lean4_utils import get_list_lean4_all_mgs_and_error_mgs
from lean_interpretor import format_error_context


# ============================================================================
# Global Lean Server Management (with auto-restart)
# ============================================================================

class LeanServerManager:
    """
    Manages a global Lean Server instance with auto-restart capability.
    Thread-safe singleton pattern.
    """
    _instance: Optional['LeanServerManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._server: Optional[Server] = None
                    cls._instance._server_lock = threading.Lock()
        return cls._instance
    
    def get_server(self) -> Server:
        """Get or create the Lean Server. Auto-restarts if needed."""
        with self._server_lock:
            if self._server is None:
                self._create_server()
            return self._server
    
    def _create_server(self):
        """Create a new Lean Server instance."""
        print("Initializing Lean Server...")
        self._server = Server(
            imports=["Init"],
            timeout=60,  # 60 seconds timeout - if exceeded, code is likely buggy
        )
        print("Lean Server initialized successfully!")
    
    def reset_server(self):
        """Force restart the Lean Server."""
        with self._server_lock:
            print("Resetting Lean Server...")
            self._server = None
            self._create_server()
    
    def _is_server_healthy(self) -> bool:
        """Check if the current server is healthy by testing with simple code."""
        if self._server is None:
            return False
        try:
            # Quick health check with simple code
            result = get_list_lean4_all_mgs_and_error_mgs("def _health_check := 42", self._server)
            return True
        except Exception:
            return False
    
    def execute_with_retry(self, lean_code: str, max_retries: int = 2) -> Dict[str, List[str]]:
        """
        Execute Lean code with automatic server restart on failure.
        
        Args:
            lean_code: The Lean code to compile
            max_retries: Maximum number of retry attempts
            
        Returns:
            Result dict with 'all_messages' and 'error_messages'
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Pre-check: ensure server is healthy before executing
                with self._server_lock:
                    if not self._is_server_healthy():
                        print(f"Server unhealthy (attempt {attempt + 1}), restarting...")
                        self._server = None
                        self._create_server()
                
                server = self.get_server()
                result = get_list_lean4_all_mgs_and_error_mgs(lean_code, server)
                return result
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Timeout errors indicate buggy code - don't retry
                if 'timeout' in error_str.lower():
                    print(f"Compilation timeout (60s) - code is likely too complex or buggy")
                    # Reset server for next request, but don't retry this one
                    self.reset_server()
                    error_msg = "Compilation timeout (60 seconds). The code is likely too complex or contains infinite loops. Please simplify."
                    return {
                        'all_messages': [error_msg],
                        'error_messages': [error_msg]
                    }
                
                # Check if this is a server crash (not a normal Lean error)
                crash_keywords = [
                    'server', 'process', 'eof', 'broken pipe', 'connection', 
                    'not running', 'assert', 'crashed', 'terminated', 'proc'
                ]
                if any(keyword in error_str.lower() for keyword in crash_keywords):
                    print(f"Server error detected (attempt {attempt + 1}): {error_str[:200]}")
                    if attempt < max_retries:
                        print("Attempting to restart server...")
                        self.reset_server()
                        continue
                
                # For other errors, don't retry
                break
        
        # If all retries failed, return error as message
        error_msg = f"Server error after {max_retries + 1} attempts: {last_error}"
        return {
            'all_messages': [error_msg],
            'error_messages': [error_msg]
        }


# Global server manager instance
server_manager = LeanServerManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    print("Starting up Lean Feedback Server...")
    print("Using persistent Lean Server with auto-restart capability.")
    # Pre-initialize the server
    server_manager.get_server()
    yield
    print("Shutting down Lean Feedback Server...")


# Request/Response models
class LeanCodeRequest(BaseModel):
    """Request model for Lean code evaluation."""
    lean_code: str
    remove_import_errors: bool = True  # Whether to auto-remove import errors


class FeedbackResponse(BaseModel):
    """Response model for feedback results."""
    score: float
    feedback: str
    valid: bool
    num_errors: int
    error_messages: list[str]
    error_details: list[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str


# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Lean Feedback Server",
    description="A web server for evaluating Lean 4 code with persistent server and auto-restart",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def lean_interpreter_with_server(lean_code: str, retry_on_crash: bool = True) -> Dict[str, Any]:
    """
    Run Lean interpreter using the managed global server.
    Similar to lean_interpretor.lean_interpreter but uses persistent server.
    """
    # Use the server manager (with auto-restart)
    result = server_manager.execute_with_retry(lean_code)
    
    all_messages: List[str] = result['all_messages']
    error_messages: List[str] = result['error_messages']
    
    # Check if error messages indicate a server crash (not a normal Lean error)
    # These errors come from lean4_utils.py which catches exceptions internally
    crash_indicators = [
        'PyPantograph server threw some exception',
        'Server not running',
        'server crashed',
        'AssertionError',
        'traceback',
    ]
    
    if retry_on_crash and error_messages:
        combined_errors = ' '.join(error_messages).lower()
        if any(indicator.lower() in combined_errors for indicator in crash_indicators):
            print("Detected server crash in error messages, resetting and retrying...")
            server_manager.reset_server()
            # Retry once
            return lean_interpreter_with_server(lean_code, retry_on_crash=False)
    
    num_errors: int = len(error_messages)
    has_errors: bool = num_errors > 0

    # Process error messages to include context
    error_details = []
    code_lines = lean_code.strip().split('\n')
    
    for error_msg in error_messages:
        try:
            parts = error_msg.split(':')
            if len(parts) >= 2:
                line_num = int(parts[1])
                formatted_error = format_error_context(error_msg, code_lines, line_num)
                error_details.append(formatted_error)
            else:
                error_details.append(f"Could not parse line number from error: {error_msg}")
        except Exception as e:
            error_details.append(f"Error processing message: {error_msg}\nException: {str(e)}")

    if has_errors:
        summary = f"Lean code compilation FAILED with {num_errors} errors."
    else:
        summary = "Lean code compiled successfully with 0 errors."
            
    return {
        "valid": not has_errors,
        "all_messages": all_messages,
        "error_messages": error_messages,
        "has_errors": has_errors,
        "num_errors": num_errors,
        "summary": summary,
        "error_details": error_details
    }


def remove_import_error_with_server(lean_code: str) -> str:
    """
    Remove import errors using the managed global server.
    Similar to lean_interpretor.remove_import_error but uses persistent server.
    """
    result = lean_interpreter_with_server(lean_code)
    if result["valid"]:
        return lean_code
    
    error_messages = result["error_messages"]
    code_lines = lean_code.strip().split('\n')
    problematic_lines = set()
    
    for error_msg in error_messages:
        if "invalid 'import' command" in error_msg:
            try:
                parts = error_msg.split(':')
                if len(parts) >= 3:
                    line_num = int(parts[1])
                    if 1 <= line_num <= len(code_lines):
                        problematic_lines.add(line_num - 1)
            except:
                pass
    
    if problematic_lines:
        filtered_lines = [line for i, line in enumerate(code_lines) if i not in problematic_lines]
        modified_code = '\n'.join(filtered_lines)
        
        # Check recursively
        result = lean_interpreter_with_server(modified_code)
        if not result["valid"] and any("invalid 'import' command" in msg for msg in result["error_messages"]):
            return remove_import_error_with_server(modified_code)
        
        return modified_code
    
    return lean_code


def get_feedback(lean_code: str, remove_imports: bool = True) -> tuple[float, str, dict]:
    """
    Get feedback from Lean code using the persistent server.
    
    Uses global Lean Server with auto-restart capability for better performance.
    
    Args:
        lean_code: The Lean 4 code to evaluate
        remove_imports: Whether to automatically remove import errors
        
    Returns:
        Tuple of (score, feedback, result_dict)
    """
    try:
        # Optionally remove import errors (using persistent server)
        code_to_check = remove_import_error_with_server(lean_code) if remove_imports else lean_code
        
        # Run the Lean interpreter (using persistent server)
        result = lean_interpreter_with_server(code_to_check)
        correctness = result["valid"]
        score = 1.0 if correctness else 0.0

        if correctness:
            feedback = "The answer is correct! No need to change anything."
        else:
            num_errors = result["num_errors"]
            
            # Get error details with context if available, otherwise raw messages
            error_details = result.get("error_details", result["error_messages"])
            errors_str = "\n\n".join(error_details)
            
            # Build feedback
            feedback = f"""Lean compilation FAILED with {num_errors} errors.

Errors:
{errors_str}
"""
        return score, feedback, result

    except Exception as e:
        error_str = str(e)
        print(f"Lean compilation error: {error_str}")
        
        # Return error as feedback
        return 0.0, f"Error occurred: {error_str}. Please fix the error and try again.", {
            "valid": False,
            "num_errors": 1,
            "error_messages": [error_str],
            "error_details": [error_str],
        }


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="ok",
        message="Lean Feedback Server is running. POST to /feedback with your Lean code."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Server is healthy")


@app.post("/reset", response_model=HealthResponse)
async def reset_lean_server():
    """
    Force restart the Lean Server.
    Use this if the server gets stuck or behaves unexpectedly.
    """
    try:
        server_manager.reset_server()
        return HealthResponse(status="ok", message="Lean Server has been reset successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset server: {str(e)}")




@app.post("/feedback", response_model=FeedbackResponse)
async def evaluate_lean_code(request: LeanCodeRequest):
    """
    Evaluate Lean 4 code and return feedback.
    
    Args:
        request: LeanCodeRequest containing the Lean code to evaluate
        
    Returns:
        FeedbackResponse with score, feedback, and detailed error information
    """
    if not request.lean_code or not request.lean_code.strip():
        raise HTTPException(status_code=400, detail="lean_code cannot be empty")
    
    try:
        score, feedback, result = get_feedback(
            request.lean_code, 
            remove_imports=request.remove_import_errors
        )
        
        return FeedbackResponse(
            score=score,
            feedback=feedback,
            valid=result.get("valid", False),
            num_errors=result.get("num_errors", 0),
            error_messages=result.get("error_messages", []),
            error_details=result.get("error_details", []),
        )
    
    except RuntimeError as e:
        # System errors (asyncio conflicts, etc.)
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def main():
    """Main function to run the server."""
    parser = argparse.ArgumentParser(description="Lean Feedback Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting Lean Feedback Server on {args.host}:{args.port}")
    print(f"API docs available at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "lean_feedback_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

