//! Sample inference demo that exercises various tensor operations and generates an HTML trace.
//!
//! Run with: `cargo run --example sample_inference`

use burn::backend::Wgpu;
use burn::prelude::*;
use burn_tracing_backend::{Profiler, finish_tracing, start_tracing, write_trace};


type B = Profiler<Wgpu>;

/// A simple MLP-like forward pass to demonstrate the profiler.
fn sample_mlp_forward(device: &<B as Backend>::Device) {
    // Create input: batch=4, features=16
    let input: Tensor<B, 2> = Tensor::random([4, 16], burn::tensor::Distribution::Normal(0.0, 1.0), device);

    // Layer 1: linear (matmul + bias) + relu
    let w1: Tensor<B, 2> = Tensor::random([16, 32], burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let b1: Tensor<B, 1> = Tensor::zeros([32], device);
    let h1 = input.matmul(w1).add(b1.unsqueeze());
    let h1 = h1.clamp_min(0.0); // ReLU via clamp

    // Layer 2: linear + sigmoid
    let w2: Tensor<B, 2> = Tensor::random([32, 8], burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let b2: Tensor<B, 1> = Tensor::zeros([8], device);
    let h2 = h1.matmul(w2).add(b2.unsqueeze());
    let h2 = burn::tensor::activation::sigmoid(h2);

    // Layer 3: reduction ops
    let _mean = h2.clone().mean();
    let _sum = h2.clone().sum();
    let _max = h2.clone().max();

    // Sync to force execution
    B::sync(device).unwrap();
}

/// Demonstrate element-wise ops, comparisons, and data movement.
fn sample_elementwise_ops(device: &<B as Backend>::Device) {
    let a: Tensor<B, 2> = Tensor::random([8, 8], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
    let b: Tensor<B, 2> = Tensor::random([8, 8], burn::tensor::Distribution::Uniform(0.0, 1.0), device);

    // Arithmetic chain
    let c = a.clone().add(b.clone());
    let d = c.mul(a.clone());
    let e = d.sub(b.clone());
    let f = e.div(a.clone().add_scalar(1e-6));

    // Math functions
    let _g = f.clone().exp();
    let _h = f.clone().log();
    let _i = f.clone().abs().sqrt();
    let _j = f.clone().sin();
    let _k = f.cos();

    // Comparisons
    let _mask = a.clone().greater(b.clone());

    // Reshape and transpose
    let reshaped = a.clone().reshape([2, 4, 8]);
    let _transposed = reshaped.swap_dims(1, 2);

    // Sync
    B::sync(device).unwrap();
}

/// Demonstrate some integer and bool operations.
fn sample_int_bool_ops(device: &<B as Backend>::Device) {
    let a: Tensor<B, 1, Int> = Tensor::arange(0..16, device);
    let b: Tensor<B, 1, Int> = Tensor::arange(16..32, device);

    let _sum = a.clone().add(b);
    let _reshaped = a.reshape([4, 4]);

    B::sync(device).unwrap();
}

fn main() {
    let device = Default::default();

    start_tracing();

    println!("Running sample MLP forward pass...");
    sample_mlp_forward(&device);

    println!("Running element-wise operations...");
    sample_elementwise_ops(&device);

    println!("Running int/bool operations...");
    sample_int_bool_ops(&device);

    let events = finish_tracing();

    println!("\nTrace summary:");
    println!("  Total operations: {}", events.len());
    println!(
        "  Sync points: {}",
        events.iter().filter(|e| e.category == burn_tracing_backend::OpCategory::Sync).count()
    );

    let total_us: f64 = events.iter().map(|e| e.duration_us).sum();
    println!("  Total wall time: {:.1}us", total_us);

    let path = "trace_data.js";
    write_trace(&events, path).unwrap();
    println!("\nTrace written to: {}", path);
    println!("Place next to visualiser.html and serve via localhost.");
}
