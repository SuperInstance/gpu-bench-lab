#!/usr/bin/env python3
"""Experiment 1: Packed Ternary Representations
How does 2-bit packing actually perform vs alternatives?"""
import torch, time, numpy as np

def bench(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), np.mean(times)

print("EXPERIMENT 1: Packed Ternary Representations")
print("=" * 60)
print()

sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

for m, k, n in sizes:
    print(f"--- {m}x{k}x{n} ---")
    a_rand = torch.randn(m, k, device='cuda')
    b_rand = torch.randn(k, n, device='cuda')
    
    # 1. FP32 baseline
    a32 = a_rand.clone(); b32 = b_rand.clone()
    med, avg = bench(lambda: a32 @ b32)
    mem = m*k*4 + k*n*4 + m*n*4
    print(f"  FP32:          {avg:7.2f}ms  {mem/1e6:.1f}MB")
    
    # 2. FP16
    a16 = a_rand.half(); b16 = b_rand.half()
    med, avg = bench(lambda: a16 @ b16)
    mem16 = m*k*2 + k*n*2 + m*n*2
    print(f"  FP16:          {avg:7.2f}ms  {mem16/1e6:.1f}MB  ({mem/mem16:.1f}x density)")
    
    # 3. Sign ternarize {-1, +1} then FP16 matmul
    a_sign = torch.sign(a_rand).half(); b_sign = torch.sign(b_rand).half()
    med, avg = bench(lambda: a_sign @ b_sign)
    print(f"  Sign->FP16:    {avg:7.2f}ms")
    
    # 4. Ternary {-1,0,+1} direct as FP16
    a_tern = torch.where(a_rand > 0.33, 1.0, torch.where(a_rand < -0.33, -1.0, 0.0)).half()
    b_tern = torch.where(b_rand > 0.33, 1.0, torch.where(b_rand < -0.33, -1.0, 0.0)).half()
    med, avg = bench(lambda: a_tern @ b_tern)
    print(f"  Ternary FP16:  {avg:7.2f}ms")
    
    # 5. Packed int8 (simulate 2-bit packing via int8)
    a_int = a_tern.to(torch.int8); b_int = b_tern.to(torch.int8)
    # PyTorch doesn't support int8 matmul directly, use float
    med, avg = bench(lambda: a_int.float() @ b_int.float())
    print(f"  Int8->FP32:    {avg:7.2f}ms")
    
    # 6. Binary (1-bit) XNOR matmul
    a_bin = (a_rand > 0).half() * 2 - 1
    b_bin = (b_rand > 0).half() * 2 - 1
    med, avg = bench(lambda: a_bin @ b_bin)
    print(f"  Binary ±1:     {avg:7.2f}ms")
    
    # 7. Sparse ternary (lots of zeros)
    a_sparse = torch.where(a_rand.abs() > 0.8, torch.sign(a_rand), torch.zeros_like(a_rand)).half()
    b_sparse = torch.where(b_rand.abs() > 0.8, torch.sign(b_rand), torch.zeros_like(b_rand)).half()
    med, avg = bench(lambda: a_sparse @ b_sparse)
    sparsity = (a_sparse == 0).float().mean().item()
    print(f"  Sparse ({sparsity:.0%} zeros): {avg:7.2f}ms")
    print()

# Memory bandwidth experiment
print("\n--- Memory Bandwidth: Ternary Packing ---")
print("Moving data to/from GPU is often the bottleneck, not compute.")
print()
for m, k, n in [(1024, 1024, 1024), (4096, 4096, 4096)]:
    a32 = torch.randn(m, k, device='cuda')
    b32 = torch.randn(k, n, device='cuda')
    a16 = a32.half(); b16 = b32.half()
    
    # Transfer time: CPU -> GPU
    a_cpu = a32.cpu()
    _, transfer_fp32 = bench(lambda: a_cpu.to('cuda'))
    a_cpu16 = a16.cpu()
    _, transfer_fp16 = bench(lambda: a_cpu16.to('cuda'))
    
    print(f"  {m}x{k}: FP32 transfer {transfer_fp32:.2f}ms ({m*k*4/1e6:.1f}MB) | FP16 transfer {transfer_fp16:.2f}ms ({m*k*2/1e6:.1f}MB)")
    print(f"          If 2-bit packed: theoretical {m*k*0.25/1e6:.1f}MB, {transfer_fp32/(m*k*4)*m*k*0.25:.2f}ms transfer")
    print()
