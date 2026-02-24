"""
Test different methods of calling LLM APIs:
1. Using Opto's LLM wrapper (via LiteLLM proxy)
2. Direct LiteLLM SDK call
3. Direct Anthropic SDK call
4. OpenAI SDK call to LiteLLM proxy
"""

import secrets_local
from opto.utils.llm import LLM
from opto.optimizers.utils import print_color 
import os


def test_opto_llm_wrapper():
    """Test 1: Using Opto's LLM wrapper with CustomLLM backend (via LiteLLM proxy)"""
    print("\n" + "="*80)
    print("TEST 1: Opto LLM Wrapper (CustomLLM backend via LiteLLM Proxy)")
    print("="*80)
    
    llm = LLM()  # Uses CustomLLM backend from secrets_local.py
    
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello from Opto LLM!' in one sentence."}
    ]
    
    response = llm(messages=prompt_messages)
    print_color(f"Response: {response.choices[0].message.content}", color="green")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")


def test_litellm_direct():
    """Test 2: Direct LiteLLM SDK call"""
    print("\n" + "="*80)
    print("TEST 2: Direct LiteLLM SDK Call")
    print("="*80)
    
    import litellm
    
    # When calling LiteLLM proxy directly, we need to use the openai/ prefix
    # because the proxy exposes an OpenAI-compatible endpoint
    response = litellm.completion(
        model="openai/claude-3.7-sonnet",  # Use openai/ prefix for proxy
        messages=[
            {"role": "user", "content": "Say 'Hello from LiteLLM!' in one sentence."}
        ],
        api_base="http://localhost:30000",
        api_key="sk-1234"
    )
    
    print_color(f"Response: {response.choices[0].message.content}", color="blue")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")


def test_anthropic_direct():
    """Test 3: Direct Anthropic SDK call (via LiteLLM proxy configured as Anthropic endpoint)"""
    print("\n" + "="*80)
    print("TEST 3: Direct Anthropic SDK Call (via LiteLLM Proxy)")
    print("="*80)
    
    try:
        from anthropic import Anthropic
    except ImportError:
        print_color("⚠️  Anthropic SDK not installed. Install with: pip install anthropic", color="yellow")
        print("Skipping this test...")
        return
    
    # Use the LiteLLM proxy as Anthropic endpoint
    client = Anthropic(
        api_key="sk-1234",  # LiteLLM proxy API key
        base_url="http://localhost:30000"  # LiteLLM proxy URL
    )
    
    response = client.messages.create(
        model="claude-3.7-sonnet",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Say 'Hello from Anthropic SDK!' in one sentence."}
        ]
    )
    
    print_color(f"Response: {response.content[0].text}", color="cyan")
    print(f"Model: {response.model}")
    print(f"Tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}")


def test_openai_sdk_to_litellm():
    """Test 4: OpenAI SDK call to LiteLLM proxy (OpenAI-compatible endpoint)"""
    print("\n" + "="*80)
    print("TEST 4: OpenAI SDK Call to LiteLLM Proxy")
    print("="*80)
    
    from openai import OpenAI
    
    client = OpenAI(
        api_key="sk-1234",
        base_url="http://localhost:30000"
    )
    
    response = client.chat.completions.create(
        model="claude-3.7-sonnet",
        messages=[
            {"role": "user", "content": "Say 'Hello from OpenAI SDK!' in one sentence."}
        ]
    )
    
    print_color(f"Response: {response.choices[0].message.content}", color="yellow")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")


def test_all():
    """Run all tests"""
    print("\n" + "🚀 " + "="*76 + " 🚀")
    print("Running All LLM API Tests")
    print("🚀 " + "="*76 + " 🚀")
    
    try:
        test_opto_llm_wrapper()
    except Exception as e:
        print_color(f"❌ Test 1 failed: {e}", color="red")
    
    try:
        test_litellm_direct()
    except Exception as e:
        print_color(f"❌ Test 2 failed: {e}", color="red")
    
    try:
        test_anthropic_direct()
    except Exception as e:
        print_color(f"❌ Test 3 failed: {e}", color="red")
    
    try:
        test_openai_sdk_to_litellm()
    except Exception as e:
        print_color(f"❌ Test 4 failed: {e}", color="red")
    
    print("\n" + "✅ " + "="*76 + " ✅")
    print("All Tests Completed!")
    print("✅ " + "="*76 + " ✅\n")


if __name__ == "__main__":
    # Run all tests by default
    test_all()
    
    # Or run individual tests:
    # test_opto_llm_wrapper()
    # test_litellm_direct()
    # test_anthropic_direct()
    # test_openai_sdk_to_litellm()