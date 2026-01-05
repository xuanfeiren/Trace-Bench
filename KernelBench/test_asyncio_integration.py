"""
Test using asyncio with eval_single_sample_modal.remote()
Run with: modal run test_asyncio_integration.py
"""

import asyncio
import time
from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping
from datasets import load_dataset


@app.local_entrypoint()
def test_asyncio():
    """
    Main test function that demonstrates asyncio integration.
    """
    print("=" * 60)
    print("Testing Asyncio Integration with Modal GPU Evaluation")
    print("=" * 60)

    # Load test data
    print("\nLoading test data...")
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()
    gpu_arch = gpu_arch_mapping["L40S"]

    # Define async functions inside this main function
    async def test_direct_call():
        """Test 1: Call .remote() directly from async context"""
        print("\n[Test 1] Calling .remote() directly from async function...")
        start = time.time()
        result = eval_single_sample_modal.remote(
            ref_arch_src, custom_cuda, False, gpu_arch,
            num_correct_trials=3, num_perf_trials=20
        )
        elapsed = time.time() - start
        print(f"  ✓ Result: correctness={result['correctness']}, runtime={result['runtime']:.2f}ms")
        print(f"  ✓ Wall-clock time: {elapsed:.2f}s")
        print(f"  Note: This works fine - Modal handles the sync call")
        return result

    async def test_with_to_thread():
        """Test 2: Use asyncio.to_thread() to avoid blocking"""
        print("\n[Test 2] Using asyncio.to_thread() for non-blocking execution...")
        start = time.time()
        result = await asyncio.to_thread(
            eval_single_sample_modal.remote,
            ref_arch_src, custom_cuda, False, gpu_arch,
            num_correct_trials=3, num_perf_trials=20
        )
        elapsed = time.time() - start
        print(f"  ✓ Result: correctness={result['correctness']}, runtime={result['runtime']:.2f}ms")
        print(f"  ✓ Wall-clock time: {elapsed:.2f}s")
        print(f"  Note: Doesn't block event loop - other async tasks can run")
        return result

    async def test_parallel_with_asyncio():
        """Test 3: Run multiple evals in parallel with asyncio"""
        print("\n[Test 3] Running 3 evaluations in parallel with asyncio.gather()...")

        async def run_eval(eval_id):
            print(f"  Starting eval {eval_id}...")
            result = await asyncio.to_thread(
                eval_single_sample_modal.remote,
                ref_arch_src, custom_cuda, False, gpu_arch,
                num_correct_trials=3, num_perf_trials=20
            )
            print(f"  ✓ Eval {eval_id} complete: {result['runtime']:.2f}ms")
            return result

        start = time.time()
        results = await asyncio.gather(run_eval(1), run_eval(2), run_eval(3))
        elapsed = time.time() - start

        print(f"  ✓ All complete! Runtimes: {[f'{r['runtime']:.2f}ms' for r in results]}")
        print(f"  ✓ Total wall-clock time: {elapsed:.2f}s")
        return results

    async def test_mixed_async_work():
        """Test 4: Mix GPU eval with other async operations"""
        print("\n[Test 4] Mixing GPU eval with other async work...")

        async def simulate_async_io(name, duration):
            print(f"  [{name}] Starting async I/O ({duration}s)...")
            await asyncio.sleep(duration)
            print(f"  [{name}] ✓ Done")
            return f"Result from {name}"

        start = time.time()

        # Run GPU eval and async I/O concurrently
        gpu_task = asyncio.to_thread(
            eval_single_sample_modal.remote,
            ref_arch_src, custom_cuda, False, gpu_arch,
            num_correct_trials=3, num_perf_trials=20
        )
        io_task1 = simulate_async_io("API-Call", 2.0)
        io_task2 = simulate_async_io("DB-Query", 1.5)

        gpu_result, io_result1, io_result2 = await asyncio.gather(
            gpu_task, io_task1, io_task2
        )

        elapsed = time.time() - start
        print(f"  ✓ GPU result: {gpu_result['runtime']:.2f}ms")
        print(f"  ✓ Async results: {io_result1}, {io_result2}")
        print(f"  ✓ Total wall-clock time: {elapsed:.2f}s")
        print(f"  Note: All tasks ran concurrently!")
        return gpu_result

    # Run all async tests
    print("\n" + "-" * 60)
    asyncio.run(test_direct_call())

    print("\n" + "-" * 60)
    asyncio.run(test_with_to_thread())

    print("\n" + "-" * 60)
    asyncio.run(test_parallel_with_asyncio())

    print("\n" + "-" * 60)
    asyncio.run(test_mixed_async_work())

    print("\n" + "=" * 60)
    print("Summary:")
    print("✓ eval_single_sample_modal.remote() works from async functions")
    print("✓ Use asyncio.to_thread() for non-blocking execution")
    print("✓ Can mix GPU evals with other async operations")
    print("=" * 60)
