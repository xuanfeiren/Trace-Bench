"""Quick test script to verify Claude models work via litellm."""
import os
import sys
import importlib.util

# Load secrets_local directly (avoid pulling in the full package)
secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'secrets_local.py')
if os.path.exists(secrets_path):
    spec = importlib.util.spec_from_file_location("secrets_local", secrets_path)
    secrets_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(secrets_mod)

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

MODELS_TO_TEST = [
    "claude-3-5-sonnet-20241022",       # retired Oct 28 2025
    "claude-3.5-sonnet",                # litellm alias
    "claude-sonnet-4-5-20250929",       # Anthropic's recommended replacement
    "claude-3-7-sonnet-20250219",       # deprecated, retirement Feb 19 2026
]

def test_model(model_name: str):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )
        content = response.choices[0].message.content
        print(f"  SUCCESS: {content}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

if __name__ == "__main__":
    print(f"ANTHROPIC_API_KEY set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")

    results = {}
    for model in MODELS_TO_TEST:
        results[model] = test_model(model)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {model}")
