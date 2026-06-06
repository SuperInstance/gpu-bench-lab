# gpu-bench-lab

*Real benchmarks on real metal. RTX 4050 Laptop, 6GB VRAM, 20 SMs.*

Every number on this page came from running code on an NVIDIA GeForce RTX 4050 Laptop GPU. No theory. No estimates. Actual hardware, actual timing.

## Results Summary

### Matrix Multiply: Precision Comparison

| Operation | 256² | 1024² | 2048² | FP16 Speedup |
|-----------|------|-------|-------|-------------|
| FP32 matmul | 0.05ms | 0.40ms | 2.50ms | baseline |
| FP16 matmul | 0.04ms | 0.17ms | 0.83ms | **3.0×** |
| BF16 matmul | 0.04ms | 0.16ms | 0.81ms | **3.1×** |

**Takeaway:** FP16/BF16 give a consistent 3× speedup over FP32 on Ada Lovelace. BF16 matches FP16 but with better dynamic range. Use BF16 for training, FP16 for inference.

### Ternary Operations

| Operation | 256² | 1024² |
|-----------|------|-------|
| Sign matmul (ternarize → matmul) | 0.06ms | 0.42ms |
| Direct ternary matmul | 0.04ms | 0.39ms |
| FP32 matmul (same size) | 0.05ms | 0.40ms |

**Takeaway:** Ternary sign matmul is NOT faster than FP32 on this GPU — the tensor cores optimize FP32 so well that the overhead of sign() negates the theoretical 16× density advantage. Direct ternary values as float match FP32 speed. The win is in *memory bandwidth* (16× less data moved), not compute.

### Vector Search: GPU vs NumPy

| Scale | GPU | NumPy | Speedup |
|-------|-----|-------|---------|
| 10K × 32d | 0.08ms | 0.03ms | 0.4× (CPU wins!) |
| 100K × 32d | 0.20ms | 0.22ms | ~1× (tie) |
| 10K × 384d | 0.10ms | 0.23ms | **2.3×** |
| 100K × 384d | 1.13ms | 2.73ms | **2.4×** |

**Takeaway:** GPU wins when dimensions are high (384d) or scale is large (100K+). For small vectors (32d) and small databases (<100K), NumPy on CPU is actually FASTER due to PCIe transfer overhead. **For musician-soul's 32-dim embeddings with <10K patterns, CPU is the right choice.**

### Embedding Search (musician-soul pattern)

| Patterns | 100 queries, 32d |
|----------|-------------------|
| 1,000 | 0.14ms |
| 10,000 | 0.31ms |
| 100,000 | 1.68ms |

**Takeaway:** Even 100K patterns with 100 simultaneous queries takes 1.68ms. That's sub-2ms for a full vector DB scan. No index needed at this scale — brute force is fast enough.

### Sort: GPU Wins Big at Scale

| Size | GPU | NumPy | Speedup |
|------|-----|-------|---------|
| 100K | 0.22ms | 0.23ms | 1.0× |
| 1M | 0.41ms | 2.42ms | **5.9×** |
| 10M | 6.27ms | 34.34ms | **5.5×** |

**Takeaway:** GPU sort dominates at 1M+ elements. 5-6× faster than NumPy. For ranking harmony scores across thousands of jam sessions, GPU is the clear choice.

### Reduction: GPU Wins at Scale

| Size | GPU | NumPy | Speedup |
|------|-----|-------|---------|
| 100K | 0.06ms | 0.02ms | 0.3× (CPU wins) |
| 1M | 0.04ms | 0.11ms | **3.1×** |
| 10M | 0.25ms | 1.66ms | **6.7×** |

**Takeaway:** Same pattern — GPU overhead makes small operations slower, but at 1M+ elements, the GPU wins by 3-7×.

## The Rules of Thumb

On an RTX 4050 Laptop:

1. **FP16/BF16 always** for matmul — 3× faster, no accuracy loss for inference
2. **Ternary doesn't save compute** on tensor cores — but saves 16× memory bandwidth
3. **CPU wins for small vector search** (<100K × 32d) — PCIe overhead kills GPU advantage
4. **GPU wins for large-scale sort/reduce** (1M+ elements) — 5-7× faster
5. **cuDNN convolution needs matching versions** — otherwise skip
6. **musician-soul should stay CPU** at current scale — 100K patterns in 1.68ms is plenty fast
7. **agent-riff harmony scoring should use GPU** — ranking across thousands of sessions benefits from GPU sort

## Component Assignment

Based on these results, here's what goes where:

| Component | Best Backend | Why |
|-----------|-------------|-----|
| PatternVectorDB query (32d, <10K) | CPU (Rust) | GPU overhead > compute |
| PatternVectorDB query (384d, 100K+) | GPU (PyTorch) | 2.4× faster |
| Ternary matmul (jam scoring) | GPU (FP16) | Tensor cores, 3× faster |
| Harmony reduction (1000+ sessions) | GPU (PyTorch) | 5-6× faster sort |
| Embedding generation | GPU (PyTorch) | matmul-heavy |
| Intent extraction | CPU (regex) | Sub-ms, no GPU needed |
| KNN for fleet discovery | CPU if <100K, GPU if >100K | Scale-dependent |

## Hardware

```
NVIDIA GeForce RTX 4050 Laptop GPU
  6 GB GDDR6 VRAM
  20 SMs, Compute 8.9 (Ada Lovelace)
  PyTorch 2.12.0+cu130
  NumPy 2.2.6
```

## Running

```bash
# Rust CPU benchmarks
cd gpu-bench-lab && cargo test

# Python GPU benchmarks
python3 benchmarks/gpu_bench.py
```

## Connection to Ecosystem

- **musician-soul**: Embedding search benchmarks tell us CPU is right for 32d patterns
- **character-encounter**: Perception checks = regex = CPU, always
- **ternary-cuda-kernels**: Ternary matmul on GPU is memory-bound, not compute-bound
- **pincher**: Intent extraction = regex = sub-ms CPU, no GPU needed
- **lever-runner**: Sandbox execution = CPU, but embedding model could use GPU
- **agent-riff**: Harmony scoring at fleet scale = GPU sort/reduce

## License

MIT
