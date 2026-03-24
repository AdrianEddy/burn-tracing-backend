//! Fusion-aware profiling demo that captures concrete fusion events.
//!
//! This example wraps at the FusionRuntime level to detect exactly when
//! fused kernels execute, their kind (elementwise, matmul, reduce), and
//! how many ops were fused.
//!
//! Run with: `cargo run --example sample_fusion --features fusion`
//!
//! To verify data capture correctness, enable trace-data and run the
//! companion `verify_trace.py` script:
//!   cargo run --example sample_fusion --features "fusion,trace-data"
//!   python verify_trace.py

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
use burn::prelude::*;
use burn_fusion::Fusion;
use burn_tracing_backend::{Profiler, finish_tracing, marker, start_tracing, write_trace};

/// `Fusion<Profiler<CubeBackend<WgpuRuntime>>>` — the Profiler sits inside
/// the Fusion layer so that `ProfiledFusionRuntime` can intercept every
/// optimization dispatch.
type B = Fusion<Profiler<CubeBackend<WgpuRuntime, f32, i32, u32>>>;

/// A simple MLP-like forward pass to demonstrate the profiler.
/// Returns clones of key intermediate tensors for verification.
fn sample_mlp_forward(device: &<B as Backend>::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let input: Tensor<B, 2> =
        Tensor::random([4, 16], burn::tensor::Distribution::Normal(0.0, 1.0), device);

    // Layer 1: linear (matmul + bias) + relu
    let w1: Tensor<B, 2> =
        Tensor::random([16, 32], burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let b1: Tensor<B, 1> = Tensor::zeros([32], device);
    let h1 = input.matmul(w1).add(b1.unsqueeze());
    let h1 = h1.clamp_min(0.0);

    // Layer 2: linear + sigmoid
    let w2: Tensor<B, 2> =
        Tensor::random([32, 8], burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let b2: Tensor<B, 1> = Tensor::zeros([8], device);
    let h2 = h1.clone().matmul(w2).add(b2.unsqueeze());
    let h2 = burn::tensor::activation::sigmoid(h2);

    // Reductions
    let _mean = h2.clone().mean();
    let _sum = h2.clone().sum();
    let _max = h2.clone().max();

    B::sync(device).unwrap();

    // Return clones — data is already computed after sync
    (h1, h2)
}

/// Demonstrate element-wise ops (should be fused together).
/// Returns the mask tensor (last output of the elementwise fusion) for verification.
fn sample_elementwise_ops(device: &<B as Backend>::Device) -> Tensor<B, 2, Bool> {
    let a: Tensor<B, 2> =
        Tensor::random([8, 8], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
    let b: Tensor<B, 2> =
        Tensor::random([8, 8], burn::tensor::Distribution::Uniform(0.0, 1.0), device);

    // Arithmetic chain — all fuseable element-wise ops
    let c = a.clone().add(b.clone());
    let d = c.mul(a.clone());
    let e = d.sub(b.clone());
    let f = e.div(a.clone().add_scalar(1e-6));

    // Math functions — also fuseable
    let _g = f.clone().exp();
    let _h = f.clone().log();
    let _i = f.clone().abs().sqrt();
    let _j = f.clone().sin();
    let _k = f.cos();

    // Comparison — last output of the elementwise fusion
    let mask = a.clone().greater(b.clone());

    // Reshape + transpose (not fuseable — break fusion chains)
    let reshaped = a.clone().reshape([2, 4, 8]);
    let _transposed = reshaped.swap_dims(1, 2);

    B::sync(device).unwrap();

    mask
}

/// Integer operations.
fn sample_int_ops(device: &<B as Backend>::Device) {
    let a: Tensor<B, 1, Int> = Tensor::arange(0..16, device);
    let b: Tensor<B, 1, Int> = Tensor::arange(16..32, device);

    let sum = a.clone().add(b);
    let _reshaped = sum.reshape([4, 4]);

    B::sync(device).unwrap();
}

fn main() {
    let device = Default::default();

    start_tracing();

    marker("start").debug("Beginning sample fusion trace").emit();

    println!("Running sample MLP forward pass...");
    let (h1, h2) = {
        let _span = marker("mlp_forward").debug("2-layer MLP: 16->32->8").span();
        sample_mlp_forward(&device)
    };

    marker("mlp_done")
        .debug(format!("h1 shape: {:?}, h2 shape: {:?}", h1.dims(), h2.dims()))
        .tensor::<B>(&h1.clone().into_primitive().tensor())
        .tensor::<B>(&h2.clone().into_primitive().tensor())
        .emit();

    println!("Running element-wise operations...");
    let mask = {
        let _span = marker("elementwise_ops").span();
        sample_elementwise_ops(&device)
    };

    println!("Running int operations...");
    {
        let _span = marker("int_ops").debug("arange + reshape").span();
        sample_int_ops(&device);
    }

    marker("end").debug("All operations complete").emit();

    let events = finish_tracing();

    println!("\nTrace summary:");
    println!("  Total events: {}", events.len());
    println!(
        "  Fusion executions: {}",
        events.iter().filter(|e| e.category == burn_tracing_backend::OpCategory::Fusion).count()
    );
    println!(
        "  Sync points: {}",
        events.iter().filter(|e| e.category == burn_tracing_backend::OpCategory::Sync).count()
    );

    for e in &events {
        if let Some(kind) = &e.fusion_kind {
            println!(
                "  [{:>6.1}us] {:?} fusion — {} ops fused, {:.1}us",
                e.start_us,
                kind,
                e.num_fused_ops.unwrap_or(0),
                e.duration_us,
            );
        }
    }

    let total_us: f64 = events.iter().map(|e| e.duration_us).sum();
    println!("\n  Total wall time: {:.1}us", total_us);

    let path = "trace_data.js";
    write_trace(&events, path).unwrap();
    println!("\nTrace written to: {}", path);

    // --- Ground truth verification ---
    // Read actual tensor data AFTER sync (fusion already executed).
    // Write as JSON so verify_trace.py can compare with trace_data.js.
    let h1_data = h1.to_data();
    let h2_data = h2.to_data();
    let mask_data = mask.to_data();

    let h1_vals: Vec<f64> = h1_data.iter::<f64>().collect();
    let h2_vals: Vec<f64> = h2_data.iter::<f64>().collect();
    let mask_vals: Vec<f64> = mask_data.iter::<f64>().collect();

    let ground_truth = serde_json::json!({
        "fusion_outputs": [
            {
                "label": "matmul_1 (h1 after relu)",
                "fusion_kind": "matmul",
                "shape": h1_data.shape.iter().collect::<Vec<_>>(),
                "data": h1_vals,
            },
            {
                "label": "matmul_2 (h2 after sigmoid)",
                "fusion_kind": "matmul",
                "shape": h2_data.shape.iter().collect::<Vec<_>>(),
                "data": h2_vals,
            },
            {
                "label": "elementwise (a.greater(b) mask)",
                "fusion_kind": "elementwise",
                "shape": mask_data.shape.iter().collect::<Vec<_>>(),
                "data": mask_vals,
            },
        ]
    });

    let gt_path = "ground_truth.json";
    std::fs::write(gt_path, serde_json::to_string_pretty(&ground_truth).unwrap()).unwrap();
    println!("Ground truth written to: {}", gt_path);
    println!("Run `python verify_trace.py` to verify data capture correctness.");
}
