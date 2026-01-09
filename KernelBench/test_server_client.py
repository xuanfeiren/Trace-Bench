#!/usr/bin/env python3
"""
Test script to verify server/client setup works.
Run this AFTER starting the server.

Usage:
    1. Terminal 1: python cuda_eval_server.py --cuda-devices cuda:0 --port 6000 --host 127.0.0.1
    2. Terminal 2: python test_server_client.py
"""

from cuda_eval_client import CUDAEvalClient
from datasets import load_dataset

def main():
    print("="*60)
    print("Testing KernelBench Server/Client Setup")
    print("="*60)

    # Step 1: Connect to server
    # Try both localhost and 127.0.0.1 to handle different network configurations
    SERVER_URLS = ["http://127.0.0.1:6000", "http://localhost:6000"]

    client = None

    for url in SERVER_URLS:
        print(f"\n[1/5] Trying to connect to server at {url}...")
        test_client = CUDAEvalClient(url)
        if test_client.health_check():
            client = test_client
            print(f"✓ Connected successfully to {url}!")
            break

    # Step 2: Health check
    if client is None:
        print("\n[2/5] Testing connection...")
        print("❌ ERROR: Cannot connect to server!")
        print("\nMake sure server is running:")
        print("  python cuda_eval_server.py --cuda-devices cuda:0 cuda:1 cuda:2 cuda:3 --port 6000")
        print("\nTried connecting to:")
        for url in SERVER_URLS:
            print(f"  - {url}")
        return

    print("✓ Connected successfully!")

    # Step 3: Get server status
    print("\n[3/5] Checking server status...")
    status = client.get_server_status()
    print(f"  Server status: {status['server_status']}")
    print(f"  Available devices: {list(status['device_status'].keys())}")
    print(f"  Device status: {status['device_status']}")
    print(f"  Queued jobs: {status['queue_info']['queued_jobs']}")
    print(f"  Active jobs: {status['queue_info']['active_jobs']}")

    # Step 4: Load test data
    print("\n[4/5] Loading test data from KernelBench...")
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ref_arch_src = examples[0]['ref_arch_src']
    print(f"✓ Loaded reference implementation")

    # Simple test CUDA kernel
    custom_cuda = """
__global__ void test_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}
"""
    print(f"✓ Test kernel: multiply by 2")

    # Step 5: Submit evaluation
    print("\n[5/5] Submitting evaluation to server...")
    job_id = client.submit_evaluation(
        problem_id=1,
        sample_id=0,
        custom_cuda=custom_cuda,
        ref_arch_src=ref_arch_src,
        level=1
    )
    print(f"✓ Job submitted! Job ID: {job_id}")

    # Wait for result
    print("\nWaiting for evaluation to complete...")
    print("(This may take 1-2 minutes for first run)")
    result = client.wait_for_job(job_id, timeout=300)

    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Job ID:       {result['job_id']}")
    print(f"Status:       {result['status']}")
    print(f"Device Used:  {result['device']}")
    print("-"*60)
    print(f"Compiled:     {result['result']['compiled']}")
    print(f"Correctness:  {result['result']['correctness']}")
    print(f"Runtime:      {result['result']['runtime']:.6f} ms")
    print("="*60)

    # Success message
    if result['result']['compiled'] and result['result']['correctness']:
        print("\n🎉 SUCCESS! Server/Client setup is working correctly!")
    elif result['result']['compiled']:
        print("\n⚠️  Kernel compiled but failed correctness tests")
    else:
        print("\n❌ Kernel failed to compile")

    print("\nYou can now use this setup for your optimization workflow!")


if __name__ == "__main__":
    main()
