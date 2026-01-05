# KernelBench: Detailed Dataset and Task Analysis

## Overview

**KernelBench** is a benchmark for evaluating Large Language Models' ability to generate optimized CUDA kernel implementations from PyTorch reference code. The task involves translating high-level PyTorch operations into efficient low-level CUDA kernels.

**Dataset**: `ScalingIntelligence/KernelBench` on HuggingFace
**Total Problems**: 270 (100 Level 1, 100 Level 2, 50 Level 3, 20 Level 4)
**Domains**: Neural network operations, loss functions, activations, architectures

---

## The Task

### Input (What the LLM Receives)

```python
# Example: Hinge Loss - PyTorch Reference Implementation
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

batch_size = 32768
input_shape = (32768,)

def get_inputs():
    return [torch.rand(batch_size, *input_shape),
            torch.randint(0, 2, (batch_size,)).float() * 2 - 1]

def get_init_inputs():
    return []
```

**Key Components Provided:**
- ✅ **Model class**: Simple PyTorch implementation (often 1-5 lines)
- ✅ **Input specifications**: Batch size, tensor shapes, dimensions
- ✅ **Data generation**: `get_inputs()` function for test data
- ✅ **Initialization**: `get_init_inputs()` if the model needs parameters

### Output (What the LLM Must Generate)

A complete CUDA-optimized implementation with:

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 1. CUDA Kernel Source Code (raw CUDA C++)
hinge_loss_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward kernel - optimized with shared memory reduction
__global__ void hinge_forward_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    float* __restrict__ out,
    const int total_elems,
    const int last_dim,
    const float inv_total) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local sum with stride
    float local_sum = 0.0f;
    for (int i = idx; i < total_elems; i += stride) {
        float y = targets[i % last_dim];
        float v = 1.0f - preds[i] * y;
        if (v > 0.0f) local_sum += v;
    }

    // Parallel reduction in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Atomically add block result
    if (tid == 0) atomicAdd(out, sdata[0] * inv_total);
}

// Backward kernel (if applicable)
__global__ void hinge_backward_kernel(...) {
    // Gradient computation
}

// PyTorch binding functions
torch::Tensor hinge_loss_forward_cuda(torch::Tensor preds, torch::Tensor targets) {
    // Input validation, memory allocation, kernel launch
    ...
}
"""

# 2. C++ Function Declarations
hinge_loss_cpp_decls = r"""
torch::Tensor hinge_loss_forward_cuda(torch::Tensor preds, torch::Tensor targets);
torch::Tensor hinge_loss_backward_cuda(torch::Tensor preds, torch::Tensor targets, double grad_out);
"""

# 3. Runtime Compilation
hinge_loss_ext = load_inline(
    name="hinge_loss_ext",
    cpp_sources=hinge_loss_cpp_decls,
    cuda_sources=hinge_loss_cuda_source,
    functions=["hinge_loss_forward_cuda", "hinge_loss_backward_cuda"],
    verbose=False
)

# 4. Autograd Integration
class HingeMeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        return hinge_loss_ext.hinge_loss_forward_cuda(predictions, targets)

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        grad_preds = hinge_loss_ext.hinge_loss_backward_cuda(
            predictions, targets, float(grad_output)
        )
        return grad_preds, None

# 5. Drop-in Replacement Model
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use custom CUDA kernel when conditions are met
        if (predictions.is_cuda and targets.is_cuda and
            predictions.dtype == torch.float32):
            return HingeMeanFunction.apply(predictions, targets)
        # Fallback to PyTorch
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

# 6. Original test data generation (unchanged)
def get_inputs():
    return [torch.rand(batch_size, *input_shape),
            torch.randint(0, 2, (batch_size,)).float() * 2 - 1]

def get_init_inputs():
    return []
```

**Required Components:**
1. ✅ **CUDA Kernels**: `__global__` functions implementing the operation
2. ✅ **C++ Bindings**: Functions to interface between Python and CUDA
3. ✅ **Runtime Compilation**: `load_inline()` to compile CUDA code on-the-fly
4. ✅ **Autograd Wrapper**: `torch.autograd.Function` for gradient computation
5. ✅ **Model Class**: `ModelNew` as drop-in replacement for original `Model`
6. ✅ **Data Generation**: Copy of original `get_inputs()` and `get_init_inputs()`

---

## Evaluation Process

### Three-Stage Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: COMPILATION                                       │
│  ────────────────────────────────────────────────────────   │
│  • Loads CUDA code via torch.utils.cpp_extension            │
│  • Checks for syntax errors, type mismatches                │
│  • Result: compiled = True/False                            │
│  • If False → STOP (overall fail)                           │
└─────────────────────────────────────────────────────────────┘
                        ↓ (if compiled)
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: CORRECTNESS                                       │
│  ────────────────────────────────────────────────────────   │
│  1. Generate test inputs using get_inputs()                 │
│  2. Run reference Model.forward(inputs)                     │
│  3. Run custom ModelNew.forward(inputs)                     │
│  4. Compare outputs: torch.allclose(ref_out, custom_out)    │
│     • Relative tolerance: 1e-5                              │
│     • Absolute tolerance: 1e-8                              │
│  5. Test backward pass (if applicable)                      │
│  • Result: correctness = True/False                         │
│  • If False → STOP (overall fail)                           │
└─────────────────────────────────────────────────────────────┘
                        ↓ (if correct)
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: PERFORMANCE                                       │
│  ────────────────────────────────────────────────────────   │
│  1. GPU warmup (3-5 runs)                                   │
│  2. Run reference implementation 100 times                  │
│     • Measure: ref_mean, ref_std, ref_median               │
│  3. Run custom implementation 100 times                     │
│     • Measure: custom_mean, custom_std, custom_median      │
│  4. Compute speedup = ref_mean / custom_mean                │
│  • Result: runtime stats                                    │
└─────────────────────────────────────────────────────────────┘
```

### Output Format: KernelExecResult

```json
{
  "compiled": true,
  "correctness": true,
  "runtime": 2.45e-05,
  "runtime_stats": {
    "mean": 2.45e-05,
    "median": 2.43e-05,
    "std": 1.2e-06,
    "min": 2.38e-05,
    "max": 2.89e-05
  },
  "metadata": {
    "speedup": 3.2,
    "device": "cuda:0"
  }
}
```

### Success Criteria

| Metric | Requirement | Type |
|--------|-------------|------|
| **Compilation** | Must compile without errors | **Binary** ✅/❌ |
| **Correctness** | Must match reference output | **Binary** ✅/❌ |
| **Performance** | Speedup over reference | **Continuous** (for ranking) |

**Overall Success** = Compilation ✅ **AND** Correctness ✅
**Performance** is measured only for successful implementations and used for ranking.

---

## Dataset Structure

### Problem Distribution

| Level | Count | Complexity | Avg Code Length | Focus Area |
|-------|-------|------------|-----------------|------------|
| **1** | 100 | Single operations | 1,276 chars | Basic kernel patterns |
| **2** | 100 | Composed operations | 1,086 chars | Operation fusion |
| **3** | 50 | Full architectures | 3,501 chars | End-to-end optimization |
| **4** | 20 | Transformer models | 720 chars | Pre-trained model inference |

### Level 1: Foundational Operations (100 problems)

**Categories:**
- **Matrix Operations** (20%): MatMul variants, transpose operations
- **Activations** (15%): ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Softmax
- **Loss Functions** (10%): Hinge, MSE, CrossEntropy, KL Divergence
- **Convolutions** (25%): Conv2D/3D with various configurations
- **Pooling** (10%): MaxPool, AvgPool, AdaptivePooling
- **Normalization** (10%): BatchNorm, LayerNorm, InstanceNorm
- **Element-wise** (10%): Add, Multiply, Power, Clamp

**Example Problems:**
```
1. 100_HingeLoss
   • Simple loss: torch.mean(torch.clamp(1 - pred * target, min=0))
   • Optimization: Parallel reduction with shared memory

2. 19_ReLU
   • torch.relu(x)
   • Optimization: Vectorized memory access (float4)

3. 1_Square_matrix_multiplication
   • C = A @ B for square matrices
   • Optimization: Tiling, shared memory, thread coalescing

4. 35_conv_standard_2D
   • Standard 2D convolution
   • Optimization: im2col + matmul vs. direct convolution
```

### Level 2: Composite Operations (100 problems)

**Characteristics:**
- Combinations of 2-4 operations
- Opportunities for kernel fusion
- More complex data flow

**Example Problems:**
```
1. 100_ConvTranspose3d_Clamp_Min_Divide
   • Pipeline: ConvTranspose3d → Clamp(min) → Divide
   • Optimization: Fuse clamp+divide into post-conv kernel

2. 45_BatchNorm_ReLU_MaxPool
   • Pipeline: BatchNorm → ReLU → MaxPool2d
   • Optimization: Fuse all three into single pass

3. 72_MultiLayer_Linear_Activation
   • Multiple linear layers with activations
   • Optimization: Batched matmul with fused activations
```

### Level 3: Neural Network Architectures (50 problems)

**Characteristics:**
- Complete neural network models
- 10-50+ operations per forward pass
- Complex control flow and residual connections

**Example Problems:**
```
1. 10_ResNet101
   • 101-layer residual network
   • Components: Bottleneck blocks, Conv+BN+ReLU, residual connections
   • Code: 4,203 characters
   • Optimization: Fused bottleneck blocks, persistent kernels

2. 25_VGG16
   • 16-layer VGG architecture
   • Sequential Conv+ReLU blocks with pooling
   • Code: 3,100+ characters
   • Optimization: Fused conv blocks, efficient memory layout

3. 40_UNet
   • U-Net for image segmentation
   • Encoder-decoder with skip connections
   • Code: 5,500+ characters
   • Optimization: Concurrent encoder/decoder paths
```

### Level 4: Pre-trained Transformers (20 problems)

**Characteristics:**
- Pre-trained models from HuggingFace
- Different model architectures and sizes
- Various batch sizes and sequence lengths

**Example Problems:**
```
1. 10_google-bigbird-roberta-base_bs1024_seq32
   • BigBird-RoBERTa model
   • Batch: 1024, Sequence: 32
   • Focus: Efficient attention mechanisms

2. 16_gpt2_bs1_seq1023
   • GPT-2 model
   • Batch: 1, Sequence: 1023
   • Focus: Long sequence optimization

3. 18_EleutherAI-gpt-neo-2p7B_bs512_seq32
   • GPT-Neo 2.7B parameters
   • Batch: 512, Sequence: 32
   • Focus: Large model inference optimization
```

**Models Included:**
- GPT-2, GPT-Neo
- BERT, RoBERTa, BigBird, ELECTRA
- BART, Reformer
- OPT (Open Pre-trained Transformer)

**Variable Parameters:**
- Batch sizes: 1, 32, 512, 1024
- Sequence lengths: 32, 256, 511, 1023, 2047, 4095

---

## Optimization Techniques by Level

### Level 1: Basic Optimizations

```cuda
// Example: ReLU with vectorized memory access

__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized load: Process 4 elements at once
    if (idx * 4 < n) {
        float4 vals = reinterpret_cast<const float4*>(input)[idx];
        vals.x = vals.x > 0.0f ? vals.x : 0.0f;
        vals.y = vals.y > 0.0f ? vals.y : 0.0f;
        vals.z = vals.z > 0.0f ? vals.z : 0.0f;
        vals.w = vals.w > 0.0f ? vals.w : 0.0f;
        reinterpret_cast<float4*>(output)[idx] = vals;
    }
}
```

**Key Techniques:**
- ✅ Coalesced memory access
- ✅ Vectorized loads/stores (float4, float2)
- ✅ Shared memory for reductions
- ✅ Optimal thread/block configuration
- ✅ Avoiding bank conflicts

### Level 2: Kernel Fusion

```cuda
// Example: Fused Conv + ReLU + BatchNorm

__global__ void fused_conv_relu_bn_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias,
    float* output, int batch, int channels, int height, int width) {

    // Single kernel computes all three operations
    // 1. Convolution
    float conv_val = compute_conv(...);

    // 2. ReLU (fused, no intermediate storage)
    float relu_val = conv_val > 0.0f ? conv_val : 0.0f;

    // 3. BatchNorm (fused, no intermediate storage)
    float bn_val = bn_weight[c] * relu_val + bn_bias[c];

    output[idx] = bn_val;
}
```

**Key Techniques:**
- ✅ Kernel fusion (multiple ops in one kernel)
- ✅ Eliminating intermediate memory transfers
- ✅ Register reuse
- ✅ Tiling for cache locality

### Level 3: Architecture-Level Optimization

```cuda
// Example: Optimized ResNet Bottleneck Block

__global__ void fused_bottleneck_kernel(
    const float* input,
    const float* conv1_w, const float* conv3_w, const float* conv5_w,
    const float* bn_params,
    float* output,
    bool has_shortcut) {

    // Entire bottleneck block in one kernel:
    // 1. Conv 1x1 (channel reduction)
    // 2. BatchNorm + ReLU
    // 3. Conv 3x3 (spatial processing)
    // 4. BatchNorm + ReLU
    // 5. Conv 1x1 (channel expansion)
    // 6. BatchNorm
    // 7. Residual addition
    // 8. Final ReLU

    // All intermediate values stay in registers
    // Minimal global memory traffic
}
```

**Key Techniques:**
- ✅ Multi-stage kernel fusion
- ✅ Persistent kernels
- ✅ Custom memory layouts
- ✅ Stream parallelism for independent blocks
- ✅ Graph-level optimization

### Level 4: Transformer Optimization

**Key Techniques:**
- ✅ FlashAttention-style attention kernels
- ✅ Fused multi-head attention
- ✅ KV cache optimization for inference
- ✅ Mixed precision (FP16/BF16)
- ✅ Tensor core utilization
- ✅ Sequence parallelism

---

## Performance Characteristics

### Typical Speedups by Level

| Level | Baseline (PyTorch) | Naive CUDA | Optimized CUDA | Expert CUDA |
|-------|-------------------|------------|----------------|-------------|
| **1** | 1.0x | 1.5-2.0x | 2.5-4.0x | 4.0-8.0x |
| **2** | 1.0x | 1.3-1.8x | 2.0-3.5x | 3.5-6.0x |
| **3** | 1.0x | 1.2-1.5x | 1.8-3.0x | 3.0-5.0x |
| **4** | 1.0x | 1.1-1.3x | 1.5-2.5x | 2.5-4.0x |

### Bottlenecks by Operation Type

**Memory-Bound Operations** (harder to optimize):
- Element-wise: ReLU, Sigmoid → Limited by memory bandwidth
- Pooling → Memory access pattern critical

**Compute-Bound Operations** (more optimization headroom):
- MatMul, Convolution → Can leverage tensor cores
- Large batch operations → Better GPU utilization

---

## Infrastructure

### Server Architecture (cuda_eval_server.py)

```
┌──────────────────────────────────────────────────────┐
│  FastAPI Server (Port 6000)                          │
├──────────────────────────────────────────────────────┤
│  Endpoints:                                          │
│  • POST /evaluate      → Submit job (async)          │
│  • GET  /job/{id}      → Check status                │
│  • POST /evaluate_sync → Submit + wait               │
│  • GET  /status        → Server/device status        │
│  • GET  /jobs          → List all jobs               │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  CUDA Device Manager                                 │
├──────────────────────────────────────────────────────┤
│  • Job Queue (FIFO)                                  │
│  • Device Pool: [cuda:0, cuda:1, cuda:2, cuda:3]    │
│  • ThreadPoolExecutor for parallel evaluation        │
│  • Active Jobs: {job_id → EvaluationJob}            │
│  • Completed Jobs: {job_id → EvaluationJob}         │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  Worker Threads (one per GPU device)                 │
├──────────────────────────────────────────────────────┤
│  1. Acquire available device                         │
│  2. Launch subprocess:                               │
│     python scripts/eval_single_example.py \          │
│       --problem-id X \                               │
│       --sample-id Y \                                │
│       --device cuda:N \                              │
│       --custom-cuda "..." \                          │
│       --ref-arch-src "..."                           │
│  3. Parse JSON result from subprocess stdout         │
│  4. Release device back to pool                      │
│  5. Store result in completed_jobs                   │
└──────────────────────────────────────────────────────┘
```

### Modal AI Architecture (evaluate_with_modal.py)

```
┌──────────────────────────────────────────────────────┐
│  Local Python Script                                 │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  Modal AI Cloud                                      │
├──────────────────────────────────────────────────────┤
│  • Docker Container: nvidia/cuda:12.4.0-devel        │
│  • GPU: L40S, H100, A100, A10G, T4                   │
│  • Python 3.10 + CUDA toolkit                        │
│  • KernelBench repo mounted at /root/KernelBench     │
├──────────────────────────────────────────────────────┤
│  @app.function(image=image, gpu="L40S")              │
│  def eval_single_sample_modal(                       │
│      ref_arch_src, custom_cuda, verbose, gpu_arch):  │
│      from src.eval import eval_kernel_against_ref    │
│      return eval_kernel_against_ref(...)             │
└──────────────────────────────────────────────────────┘
```

---

## Common Challenges

### 1. Compilation Errors

**Typical Issues:**
- Missing CUDA headers (`#include <cuda.h>`)
- Type mismatches (int vs. int64_t)
- Incorrect function signatures
- Missing extern "C" for some bindings

### 2. Correctness Failures

**Typical Issues:**
- Numerical precision errors (float vs. double)
- Race conditions in parallel reductions
- Incorrect memory indexing
- Missing synchronization (`__syncthreads()`)
- Atomic operation issues

### 3. Performance Degradation

**Typical Issues:**
- Suboptimal thread/block configuration
- Memory bank conflicts
- Uncoalesced memory access
- Too much register usage (low occupancy)
- Excessive global memory traffic

---

## Dataset Statistics Summary

```
Total Problems: 270
├── Level 1: 100 (37%)
│   ├── Average code: 1,276 chars
│   └── Categories: MatMul, Activation, Loss, Conv, Pool, Norm
├── Level 2: 100 (37%)
│   ├── Average code: 1,086 chars
│   └── Focus: Operation fusion opportunities
├── Level 3: 50 (18.5%)
│   ├── Average code: 3,501 chars
│   ├── Max: 23,170 chars (complex architecture)
│   └── Architectures: ResNet, VGG, U-Net, Inception
└── Level 4: 20 (7.5%)
    ├── Average code: 720 chars
    └── Models: GPT-2, BERT, BART, BigBird, etc.
```

---

## Key Takeaways

1. **Progressive Difficulty**: From single ops (Level 1) to full transformers (Level 4)

2. **Real-World Relevance**: All operations are used in production deep learning

3. **Multiple Optimization Dimensions**:
   - Correctness (must match PyTorch exactly)
   - Performance (speedup over baseline)
   - Code quality (compilability, robustness)

4. **Infrastructure Flexibility**:
   - Self-hosted GPU servers
   - Cloud GPU via Modal AI
   - Async job queue for efficiency

5. **Evaluation Rigor**:
   - Compilation check
   - Correctness verification
   - Performance benchmarking (100 trials)
   - Statistical metrics (mean, median, std)

6. **Challenge Scope**: Requires deep knowledge of:
   - CUDA programming
   - PyTorch internals
   - GPU architecture
   - Numerical computing
   - Parallel algorithms

This benchmark tests the full stack of GPU optimization skills needed for modern deep learning systems.
