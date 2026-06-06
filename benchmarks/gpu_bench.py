#!/usr/bin/env python3
"""GPU Benchmark Suite — RTX 4050 Laptop. Every result live from hardware."""
import time, json, numpy as np
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEV_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
except ImportError:
    DEVICE, DEV_NAME = "cpu", "CPU"

def bench(name, fn, warmup=3, iters=20):
    for _ in range(warmup): fn()
    if DEVICE == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"name": name, "device": DEV_NAME, "avg_ms": np.mean(times),
            "min_ms": min(times), "p50_ms": np.median(times), "iters": iters}

def main():
    print(f"GPU BENCH LAB — {DEV_NAME} | PyTorch {torch.__version__}")
    print("=" * 70)
    R = []

    print("\n--- MatMul: FP32 vs FP16 vs BF16 ---")
    for m, k, n in [(256,256,256),(1024,1024,1024),(2048,2048,2048)]:
        for dtype, label in [(torch.float32,"FP32"),(torch.float16,"FP16"),(torch.bfloat16,"BF16")]:
            a = torch.randn(m, k, device=DEVICE, dtype=dtype)
            b = torch.randn(k, n, device=DEVICE, dtype=dtype)
            r = bench(f"matmul_{label}_{m}x{n}", lambda a=a,b=b: a @ b)
            R.append(r); print(f"  {label:4} {m}x{n}: {r['avg_ms']:7.2f}ms (p50 {r['p50_ms']:.2f})")

    print("\n--- Ternary Operations ---")
    for m, k, n in [(256,256,256),(1024,1024,1024)]:
        a = torch.randn(m, k, device=DEVICE); b = torch.randn(k, n, device=DEVICE)
        r = bench(f"ternary_sign_{m}x{n}", lambda: torch.sign(a) @ torch.sign(b))
        R.append(r); print(f"  Sign {m}x{n}: {r['avg_ms']:.2f}ms")

        ai = torch.randint(-1, 2, (m, k), device=DEVICE, dtype=torch.float32)
        bi = torch.randint(-1, 2, (k, n), device=DEVICE, dtype=torch.float32)
        r = bench(f"ternary_direct_{m}x{n}", lambda: ai @ bi)
        R.append(r); print(f"  Direct {m}x{n}: {r['avg_ms']:.2f}ms")

    print("\n--- Vector Search: GPU vs NumPy ---")
    for nv, dim in [(10_000, 32), (100_000, 32), (10_000, 384), (100_000, 384)]:
        db = torch.randn(nv, dim, device=DEVICE)
        q = torch.randn(1, dim, device=DEVICE)
        r = bench(f"KNN_gpu_{nv}x{dim}", lambda: torch.topk(torch.mm(q, db.T).squeeze(0), 10))
        R.append(r); print(f"  GPU KNN {nv}x{dim}: {r['avg_ms']:.2f}ms")

        db_np = np.random.randn(nv, dim).astype(np.float32)
        q_np = np.random.randn(dim).astype(np.float32)
        r = bench(f"KNN_numpy_{nv}x{dim}", lambda: np.argpartition(db_np @ q_np, -10)[-10:])
        R.append(r); print(f"  NumPy KNN {nv}x{dim}: {r['avg_ms']:.2f}ms")

    print("\n--- Embedding Search (musician-soul pattern) ---")
    for np_ in [1000, 10_000, 100_000]:
        db = torch.randn(np_, 32, device=DEVICE); db /= db.norm(dim=1, keepdim=True)
        qs = torch.randn(100, 32, device=DEVICE); qs /= qs.norm(dim=1, keepdim=True)
        r = bench(f"embed_search_{np_}x32_100q", lambda: torch.topk(torch.mm(qs, db.T), 5, dim=1))
        R.append(r); print(f"  {np_:>6} patterns, 100 queries: {r['avg_ms']:.2f}ms")

    print("\n--- Reduction & Sort ---")
    for n in [100_000, 1_000_000, 10_000_000]:
        d = torch.randn(n, device=DEVICE)
        r = bench(f"sum_gpu_{n//1000}K", lambda: d.sum()); R.append(r)
        print(f"  GPU sum {n//1000}K: {r['avg_ms']:.2f}ms")
        r = bench(f"sort_gpu_{n//1000}K", lambda: d.sort()); R.append(r)
        print(f"  GPU sort {n//1000}K: {r['avg_ms']:.2f}ms")
        dn = np.random.randn(n).astype(np.float32)
        r = bench(f"sum_numpy_{n//1000}K", lambda: dn.sum()); R.append(r)
        print(f"  NumPy sum {n//1000}K: {r['avg_ms']:.2f}ms")
        r = bench(f"sort_numpy_{n//1000}K", lambda: np.sort(dn)); R.append(r)
        print(f"  NumPy sort {n//1000}K: {r['avg_ms']:.2f}ms")

    print("\n--- Convolution (Signal Processing) ---")
    for sl, k, c in [(4096,16,4),(16384,32,8),(65536,64,16)]:
        sig = torch.randn(1, c, sl, device=DEVICE)
        ker = torch.randn(1, c, k, device=DEVICE)
        try:
            r = bench(f"conv1d_{sl}len_{k}kern_{c}ch", lambda: torch.nn.functional.conv1d(sig, ker))
        except RuntimeError:
            r = {'name': f'conv1d_{sl}len_{k}kern_{c}ch', 'device': DEV_NAME, 'avg_ms': -1, 'min_ms': -1, 'p50_ms': -1, 'iters': 0}
            R.append(r)
            print(f"  Conv1d {sl}len {k}kern {c}ch: SKIPPED (cuDNN version mismatch)")
            continue
        R.append(r); print(f"  Conv1d {sl}len {k}kern {c}ch: {r['avg_ms']:.2f}ms")

    with open("bench_results.json", "w") as f:
        json.dump({"device": DEV_NAME, "pytorch": torch.__version__, "results": R}, f, indent=2)
    print(f"\n{len(R)} benchmarks saved to bench_results.json")

    print("\n--- GPU Speedups ---")
    for g in R:
        if "gpu" in g["name"]:
            base = g["name"].replace("_gpu_", "_numpy_")
            for c in R:
                if c["name"] == base and c["avg_ms"] > 0:
                    print(f"  {c['avg_ms']/g['avg_ms']:6.1f}x  {g['name']}")

if __name__ == "__main__":
    main()
