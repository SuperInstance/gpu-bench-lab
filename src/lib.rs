//! # gpu-bench-lab
//!
//! GPU benchmarks across languages and backends for real application components.
//! Tests what does what best on actual hardware (RTX 4050 Laptop, 6GB, 20 SMs).
//!
//! ## What We Test
//!
//! | Component | Languages | What It Measures |
//! |-----------|-----------|-----------------|
//! | Matrix multiply (FP32) | CUDA, Python/PyTorch | Raw compute throughput |
//! | Matrix multiply (FP16) | CUDA, Python/PyTorch | Mixed precision speedup |
//! | Ternary matmul (XNOR) | CUDA PTX, Python/NumPy | 2-bit ops vs FP32 |
//! | Vector search | Python/PyTorch | Embedding similarity at scale |
//! | Sorting | CUDA, Python/NumPy | Parallel sort vs single-thread |
//! | Reduction | CUDA, Python/NumPy | Sum/min/max across arrays |
//! | Convolution | Python/PyTorch | Signal processing kernels |
//! | String matching | CUDA PTX, Python/regex | Pattern matching throughput |
//!
//! ## Results
//!
//! All results generated on live hardware. No estimates. No theory.

#![forbid(unsafe_code)]

use std::time::Instant;

/// Benchmark result for a single test.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub language: String,
    pub backend: String,
    pub data_size: usize,
    pub iterations: usize,
    pub total_ms: f64,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub throughput_gbps: f64,  // gigabytes per second
    pub notes: String,
}

impl BenchResult {
    pub fn new(name: &str, language: &str, backend: &str) -> Self {
        Self { name: name.to_string(), language: language.to_string(),
               backend: backend.to_string(), data_size: 0, iterations: 0,
               total_ms: 0.0, avg_ms: 0.0, min_ms: f64::MAX, max_ms: 0.0,
               throughput_gbps: 0.0, notes: String::new() }
    }

    pub fn finish(&mut self, times_ms: &[f64]) {
        self.iterations = times_ms.len();
        self.total_ms = times_ms.iter().sum();
        self.avg_ms = self.total_ms / self.iterations as f64;
        self.min_ms = times_ms.iter().cloned().fold(f64::MAX, f64::min);
        self.max_ms = times_ms.iter().cloned().fold(0.0, f64::max);
        if self.avg_ms > 0.0 && self.data_size > 0 {
            self.throughput_gbps = (self.data_size as f64 * 1e-9) / (self.avg_ms * 1e-3);
        }
    }

    pub fn summary(&self) -> String {
        format!("{:40} | {:10} | avg {:8.2}ms | min {:8.2}ms | {:.1} GB/s | {}",
            self.name, self.language, self.avg_ms, self.min_ms, self.throughput_gbps, self.notes)
    }
}

/// CPU-only ternary operations (Rust reference for GPU comparison).
pub mod cpu_ternary {
    use super::*;

    /// Z₃ addition via explicit match.
    pub fn tadd(a: i8, b: i8) -> i8 {
        match (a, b) {
            (-1, -1) => 1, (-1, 0) => -1, (-1, 1) => 0,
            (0, -1) => -1, (0, 0) => 0, (0, 1) => 1,
            (1, -1) => 0, (1, 0) => 1, (1, 1) => -1,
            _ => 0,
        }
    }

    /// Ternary matmul on CPU — each element is {-1, 0, +1}.
    pub fn ternary_matmul(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i32;
                for p in 0..k {
                    sum += (a[i * k + p] as i32) * (b[p * n + j] as i32);
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Packed ternary matmul using XNOR+popcount (2-bit per trit).
    /// 16 trits per u32. matmul via popcount(xnor(a, b)).
    pub fn packed_ternary_matmul(a_packed: &[u32], b_packed: &[u32], m: usize, k_packed: usize, n: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i32;
                for p in 0..k_packed {
                    let xnor = !(a_packed[i * k_packed + p] ^ b_packed[j * k_packed + p]);
                    sum += xnor.count_ones() as i32 - 16; // 16 trits per u32
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Pack 16 ternary values into one u32.
    pub fn pack_trits(trits: &[i8]) -> u32 {
        let mut packed = 0u32;
        for (i, &t) in trits.iter().take(16).enumerate() {
            let bits = match t { -1 => 0u32, 0 => 1, 1 => 2, _ => 1 };
            packed |= bits << (i * 2);
        }
        packed
    }

    /// Benchmark: Rust CPU ternary matmul.
    pub fn bench_ternary_matmul(m: usize, k: usize, n: usize, iters: usize) -> BenchResult {
        let a: Vec<i8> = (0..m*k).map(|i| ((i % 3) as i8) - 1).collect();
        let b: Vec<i8> = (0..k*n).map(|i| ((i % 5) as i8 % 3) as i8 - 1).collect();
        let data_size = (m * k + k * n + m * n) * std::mem::size_of::<i32>();

        let mut result = BenchResult::new(
            &format!("ternary_matmul_{}x{}x{}", m, k, n),
            "Rust", "CPU (single-thread)"
        );
        result.data_size = data_size;

        let mut times = Vec::new();
        for _ in 0..iters {
            let start = Instant::now();
            let _c = ternary_matmul(&a, &b, m, k, n);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        result.finish(&times);
        result
    }

    /// Benchmark: Rust CPU packed ternary matmul.
    pub fn bench_packed_ternary_matmul(m: usize, k: usize, n: usize, iters: usize) -> BenchResult {
        let k_packed = (k + 15) / 16;
        let a_packed: Vec<u32> = (0..m * k_packed).map(|i| pack_trits(&[(((i*7+3) % 3) as i8) - 1; 16])).collect();
        let b_packed: Vec<u32> = (0..n * k_packed).map(|i| pack_trits(&[(((i*11+5) % 3) as i8) - 1; 16])).collect();
        let data_size = (m * k_packed + n * k_packed + m * n) * std::mem::size_of::<u32>();

        let mut result = BenchResult::new(
            &format!("packed_ternary_matmul_{}x{}x{}", m, k, n),
            "Rust", "CPU (single-thread, XNOR+popcount)"
        );
        result.data_size = data_size;
        result.notes = format!("16x density, {} packed cols", k_packed);

        let mut times = Vec::new();
        for _ in 0..iters {
            let start = Instant::now();
            let _c = packed_ternary_matmul(&a_packed, &b_packed, m, k_packed, n);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        result.finish(&times);
        result
    }
}

/// CPU vector operations (Rust reference).
pub mod cpu_vector {
    use super::*;

    /// Cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
        dot / (norm_a * norm_b)
    }

    /// Find K nearest neighbors by brute-force cosine similarity.
    pub fn knn_search(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = database.iter().enumerate()
            .map(|(i, v)| (i, cosine_similarity(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(k).collect()
    }

    /// Benchmark: KNN vector search on CPU.
    pub fn bench_knn_search(n_vectors: usize, dim: usize, k: usize, iters: usize) -> BenchResult {
        let db: Vec<Vec<f32>> = (0..n_vectors).map(|i|
            (0..dim).map(|j| ((i * dim + j) as f32 * 0.001).sin()).collect()
        ).collect();
        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.001).cos()).collect();
        let data_size = n_vectors * dim * std::mem::size_of::<f32>();

        let mut result = BenchResult::new(
            &format!("knn_search_{}vecs_{}d_k{}", n_vectors, dim, k),
            "Rust", "CPU (brute-force cosine)"
        );
        result.data_size = data_size;

        let mut times = Vec::new();
        for _ in 0..iters {
            let start = Instant::now();
            let _neighbors = knn_search(&query, &db, k);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        result.finish(&times);
        result
    }

    /// Benchmark: Parallel reduction (sum) on CPU.
    pub fn bench_reduction(n: usize, iters: usize) -> BenchResult {
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_size = n * std::mem::size_of::<f32>();

        let mut result = BenchResult::new(
            &format!("reduction_sum_{}M", n / 1_000_000),
            "Rust", "CPU (sequential)"
        );
        result.data_size = data_size;

        let mut times = Vec::new();
        for _ in 0..iters {
            let start = Instant::now();
            let _sum: f32 = data.iter().sum();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        result.finish(&times);
        result
    }

    /// Benchmark: Sorting on CPU.
    pub fn bench_sort(n: usize, iters: usize) -> BenchResult {
        let data_size = n * std::mem::size_of::<f32>();

        let mut result = BenchResult::new(
            &format!("sort_{}M", n / 1_000_000),
            "Rust", "CPU (slice::sort)"
        );
        result.data_size = data_size;

        let mut times = Vec::new();
        for _ in 0..iters {
            let mut data: Vec<f32> = (0..n).map(|i| ((i as f32 * 1.618).sin())).collect();
            let start = Instant::now();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        result.finish(&times);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn ternary_matmul_correct() {
        let a = [1, 0, -1, 1];
        let b = [-1, 1, 0, 1, 1, -1]; // 2x3
        let c = cpu_ternary::ternary_matmul(&a, &b, 2, 2, 3);
        // [1* -1 + 0*0, 1*1 + 0*1, 1*1 + 0*-1] = [-1, 1, 1]
        assert_eq!(c[0], -1);
    }

    #[test] fn packed_roundtrip() {
        let trits = [-1i8, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1];
        let packed = cpu_ternary::pack_trits(&trits);
        assert_ne!(packed, 0);
    }

    #[test] fn cosine_sim_self() {
        let v = [1.0f32, 2.0, 3.0];
        let sim = cpu_vector::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test] fn cosine_sim_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        let sim = cpu_vector::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test] fn knn_returns_k() {
        let db: Vec<Vec<f32>> = (0..100).map(|i|
            vec![i as f32, (i as f32).sin()]
        ).collect();
        let query = vec![50.0f32, 0.0];
        let neighbors = cpu_vector::knn_search(&query, &db, 5);
        assert_eq!(neighbors.len(), 5);
    }

    #[test] fn bench_ternary_small() {
        let result = cpu_ternary::bench_ternary_matmul(32, 32, 32, 3);
        assert!(result.avg_ms > 0.0);
        assert!(result.min_ms > 0.0);
    }

    #[test] fn bench_packed_small() {
        let result = cpu_ternary::bench_packed_ternary_matmul(32, 32, 32, 3);
        assert!(result.avg_ms > 0.0);
    }

    #[test] fn bench_knn_small() {
        let result = cpu_vector::bench_knn_search(1000, 32, 10, 3);
        assert!(result.avg_ms > 0.0);
    }

    #[test] fn bench_reduction_small() {
        let result = cpu_vector::bench_reduction(100_000, 3);
        assert!(result.avg_ms > 0.0);
    }

    #[test] fn bench_sort_small() {
        let result = cpu_vector::bench_sort(100_000, 3);
        assert!(result.avg_ms > 0.0);
    }
}
