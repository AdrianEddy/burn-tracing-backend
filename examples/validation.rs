//! Comprehensive validation example that exercises many operation types and
//! saves ground truth (shapes + data) to a JSON file for automated comparison
//! with trace_data.js.
//!
//! Run with:
//!   cargo run --example validation --features "fusion,trace-data"
//!   python validate_trace.py
//!
//! The companion `validate_trace.py` script parses both files and verifies
//! that every traced operation has correct output_shape, data_preview,
//! and fusion_outputs.

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
use burn::prelude::*;
use burn_fusion::Fusion;
use burn_tracing_backend::{Profiler, finish_tracing, marker, start_tracing, write_trace};
use serde_json::json;

type B = Fusion<Profiler<CubeBackend<WgpuRuntime, f32, i32, u32>>>;
type Dev = <B as Backend>::Device;

/// Collect a ground-truth entry: label, shape, and first N values as f64.
fn gt_float<const D: usize>(label: &str, t: &Tensor<B, D>) -> serde_json::Value {
    let data = t.to_data();
    let shape: Vec<usize> = data.shape.iter().copied().collect();
    let vals: Vec<f64> = data.iter::<f64>().collect();
    json!({ "label": label, "kind": "float", "shape": shape, "data": vals })
}

fn gt_int<const D: usize>(label: &str, t: &Tensor<B, D, Int>) -> serde_json::Value {
    let data = t.to_data();
    let shape: Vec<usize> = data.shape.iter().copied().collect();
    let vals: Vec<f64> = data.iter::<f64>().collect();
    json!({ "label": label, "kind": "int", "shape": shape, "data": vals })
}

fn gt_bool<const D: usize>(label: &str, t: &Tensor<B, D, Bool>) -> serde_json::Value {
    let data = t.to_data();
    let shape: Vec<usize> = data.shape.iter().copied().collect();
    let vals: Vec<f64> = data.iter::<f64>().collect();
    json!({ "label": label, "kind": "bool", "shape": shape, "data": vals })
}

// ─── Test blocks ────────────────────────────────────────────────────

/// Block 1: Simple standalone float ops (no fusion — Wgpu ops go through directly).
fn block_standalone_float(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    // 1. from_data
    let a: Tensor<B, 2> = Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dev);
    gt.push(gt_float("1_from_data_a", &a));

    // 2. reshape
    let b = a.clone().reshape([3, 2]);
    gt.push(gt_float("2_reshape_b", &b));

    // 3. swap_dims
    let c = b.clone().swap_dims(0, 1);
    gt.push(gt_float("3_swap_dims_c", &c));

    // 4. slice
    let d = a.clone().slice([0..1, 1..3]);
    gt.push(gt_float("4_slice_d", &d));

    // 5. cat
    let e = Tensor::cat(vec![a.clone(), a.clone()], 0);
    gt.push(gt_float("5_cat_e", &e));

    B::sync(dev).unwrap();
}

/// Block 2: Fused elementwise chain.
fn block_fused_elementwise(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let a: Tensor<B, 2> =
        Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], dev);
    let b: Tensor<B, 2> =
        Tensor::from_floats([[0.5, 1.5], [2.5, 3.5]], dev);

    // 6. add → mul → sub chain (should fuse)
    let c = a.clone().add(b.clone());
    let d = c.mul(a.clone());
    let e = d.sub(b.clone());
    gt.push(gt_float("6_fused_elem_e", &e));

    // 7. exp
    let f = e.clone().exp();
    gt.push(gt_float("7_exp_f", &f));

    // 8. comparison (bool output from float fusion)
    let mask = a.clone().greater(b.clone());
    gt.push(gt_bool("8_greater_mask", &mask));

    B::sync(dev).unwrap();
}

/// Block 3: Matmul fusion.
fn block_matmul_fusion(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let x: Tensor<B, 2> =
        Tensor::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dev);
    let w: Tensor<B, 2> =
        Tensor::from_floats([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dev);
    let bias: Tensor<B, 1> = Tensor::from_floats([0.1, 0.2, 0.3], dev);

    // 9. matmul
    let h = x.matmul(w);
    gt.push(gt_float("9_matmul_h", &h));

    // 10. add bias (fused with matmul)
    let h_bias = h.add(bias.unsqueeze());
    gt.push(gt_float("10_matmul_bias_h", &h_bias));

    // 11. clamp_min (relu-like, fused)
    let h_relu = h_bias.clamp_min(0.0);
    gt.push(gt_float("11_relu_h", &h_relu));

    // 12. mean reduction (standalone after fusion)
    let m = h_relu.clone().mean();
    gt.push(gt_float("12_mean_m", &m));

    B::sync(dev).unwrap();
}

/// Block 4: Reduction ops.
fn block_reductions(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let a: Tensor<B, 2> =
        Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dev);

    // 13. sum
    let s = a.clone().sum();
    gt.push(gt_float("13_sum_s", &s));

    // 14. sum_dim
    let sd = a.clone().sum_dim(1);
    gt.push(gt_float("14_sum_dim_sd", &sd));

    // 15. mean_dim
    let md = a.clone().mean_dim(0);
    gt.push(gt_float("15_mean_dim_md", &md));

    // 16. max
    let mx = a.clone().max();
    gt.push(gt_float("16_max_mx", &mx));

    // 17. min
    let mn = a.clone().min();
    gt.push(gt_float("17_min_mn", &mn));

    // 18. argmax
    let am = a.clone().argmax(1);
    gt.push(gt_int("18_argmax_am", &am));

    // 19. argmin
    let an = a.clone().argmin(1);
    gt.push(gt_int("19_argmin_an", &an));

    B::sync(dev).unwrap();
}

/// Block 5: Int ops (standalone, no fusion).
fn block_int_ops(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let a: Tensor<B, 1, Int> = Tensor::arange(0..8, dev);

    // 20. int reshape
    let b = a.clone().reshape([2, 4]);
    gt.push(gt_int("20_int_reshape_b", &b));

    // 21. int add
    let c: Tensor<B, 1, Int> = Tensor::arange(8..16, dev);
    let d = a.clone().add(c);
    gt.push(gt_int("21_int_add_d", &d));

    // 22. int mul_scalar
    let e = a.clone().mul_scalar(3);
    gt.push(gt_int("22_int_mul_scalar_e", &e));

    // 23. int clamp
    let f = a.clone().clamp(2, 5);
    gt.push(gt_int("23_int_clamp_f", &f));

    B::sync(dev).unwrap();
}

/// Block 6: Bool ops.
fn block_bool_ops(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let a: Tensor<B, 1, Bool> =
        Tensor::<B, 1, Int>::from_ints([1, 0, 1, 0, 1, 0], dev).bool();
    let b: Tensor<B, 1, Bool> =
        Tensor::<B, 1, Int>::from_ints([1, 1, 0, 0, 1, 1], dev).bool();

    // 24. bool not
    let c = a.clone().bool_not();
    gt.push(gt_bool("24_bool_not_c", &c));

    // 25. bool and
    let d = a.clone().bool_and(b.clone());
    gt.push(gt_bool("25_bool_and_d", &d));

    // 26. bool or
    let e = a.clone().bool_or(b.clone());
    gt.push(gt_bool("26_bool_or_e", &e));

    // 27. bool into_int
    let f = a.clone().int();
    gt.push(gt_int("27_bool_into_int_f", &f));

    B::sync(dev).unwrap();
}

/// Block 7: Math functions chain (fused).
fn block_math_chain(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let a: Tensor<B, 2> =
        Tensor::from_floats([[0.5, 1.0], [1.5, 2.0]], dev);

    // 28. sin
    let s = a.clone().sin();
    gt.push(gt_float("28_sin_s", &s));

    // 29. cos
    let c = a.clone().cos();
    gt.push(gt_float("29_cos_c", &c));

    // 30. abs + sqrt
    let sq = a.clone().abs().sqrt();
    gt.push(gt_float("30_abs_sqrt_sq", &sq));

    // 31. log (of positive values)
    let l = a.clone().log();
    gt.push(gt_float("31_log_l", &l));

    // 32. pow_scalar
    let p = a.clone().powf_scalar(2.0);
    gt.push(gt_float("32_pow_p", &p));

    B::sync(dev).unwrap();
}

/// Block 8: Transpose, permute, flip.
fn block_shape_ops(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let a: Tensor<B, 3> =
        Tensor::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dev); // [1,3,2]

    // 33. transpose (swap last two dims)
    let t = a.clone().transpose();
    gt.push(gt_float("33_transpose_t", &t));

    // 34. flip
    let f = a.clone().flip([1]);
    gt.push(gt_float("34_flip_f", &f));

    // 35. expand
    let expanded = a.clone().expand([4, 3, 2]);
    gt.push(gt_float("35_expand_expanded", &expanded));

    B::sync(dev).unwrap();
}

/// Block 9: Second matmul fusion (larger).
fn block_matmul_large(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let x: Tensor<B, 2> =
        Tensor::random([4, 8], burn::tensor::Distribution::Uniform(-1.0, 1.0), dev);
    let w: Tensor<B, 2> =
        Tensor::random([8, 16], burn::tensor::Distribution::Uniform(-1.0, 1.0), dev);

    // 36. matmul
    let h = x.matmul(w);
    gt.push(gt_float("36_matmul_large_h", &h));

    // 37. sigmoid (fused)
    let s = burn::tensor::activation::sigmoid(h);
    gt.push(gt_float("37_sigmoid_s", &s));

    // 38. reshape (standalone after fusion)
    let r = s.reshape([2, 2, 16]);
    gt.push(gt_float("38_reshape_r", &r));

    B::sync(dev).unwrap();
}

/// Block 10: Type conversions.
fn block_conversions(dev: &Dev, gt: &mut Vec<serde_json::Value>) {
    let a: Tensor<B, 1> = Tensor::from_floats([1.5, 2.7, 3.1, 4.9], dev);

    // 39. float into_int
    let i = a.clone().int();
    gt.push(gt_int("39_float_into_int_i", &i));

    // 40. int into_float
    let f = i.clone().float();
    gt.push(gt_float("40_int_into_float_f", &f));

    B::sync(dev).unwrap();
}

// ─── Main ───────────────────────────────────────────────────────────

fn main() {
    let dev: Dev = Default::default();
    let mut gt: Vec<serde_json::Value> = Vec::new();

    start_tracing();

    marker("validation_start")
        .debug("Validation suite with 40 ground-truth entries")
        .binary(b"BURN_VALIDATE_v1".to_vec())
        .emit();

    println!("Block 1: Standalone float ops");
    {
        let _span = marker("block_1_standalone_float").span();
        block_standalone_float(&dev, &mut gt);
    }

    println!("Block 2: Fused elementwise chain");
    {
        let _span = marker("block_2_fused_elementwise").span();
        block_fused_elementwise(&dev, &mut gt);
    }

    marker("phase_1_done").debug("Blocks 1-2 complete (standalone + elementwise)").emit();

    println!("Block 3: Matmul fusion");
    {
        let _span = marker("block_3_matmul").span();
        block_matmul_fusion(&dev, &mut gt);
    }

    println!("Block 4: Reductions");
    {
        let _span = marker("block_4_reductions").span();
        block_reductions(&dev, &mut gt);
    }

    println!("Block 5: Int ops");
    {
        let _span = marker("block_5_int_ops").span();
        block_int_ops(&dev, &mut gt);
    }

    println!("Block 6: Bool ops");
    {
        let _span = marker("block_6_bool_ops").span();
        block_bool_ops(&dev, &mut gt);
    }

    marker("phase_2_done").debug("Blocks 3-6 complete (matmul + reductions + int + bool)").emit();

    println!("Block 7: Math chain (fused)");
    {
        let _span = marker("block_7_math_chain").span();
        block_math_chain(&dev, &mut gt);
    }

    println!("Block 8: Shape ops");
    {
        let _span = marker("block_8_shape_ops").span();
        block_shape_ops(&dev, &mut gt);
    }

    println!("Block 9: Large matmul + sigmoid");
    {
        let mut span = marker("block_9_matmul_large").span();
        block_matmul_large(&dev, &mut gt);
        span.set_debug("4x8 @ 8x16 matmul + sigmoid + reshape");
    }

    println!("Block 10: Type conversions");
    {
        let _span = marker("block_10_conversions").span();
        block_conversions(&dev, &mut gt);
    }

    let x: Tensor<B, 2> =
        Tensor::random([4, 8], burn::tensor::Distribution::Uniform(-1.0, 1.0), &dev);
    let y: Tensor<B, 2> =
        Tensor::random([4, 8], burn::tensor::Distribution::Uniform(-1.0, 1.0), &dev);
    B::sync(&dev).unwrap();
    marker("validation_end")
        .debug(format!("{} ground-truth entries collected", gt.len()))
        .tensor::<B>(&x.into_primitive().tensor())
        .tensor::<B>(&y.into_primitive().tensor())
        .emit();

    let events = finish_tracing();

    // Print summary
    let n_fusion = events.iter().filter(|e| e.category == burn_tracing_backend::OpCategory::Fusion).count();
    let n_sync = events.iter().filter(|e| e.category == burn_tracing_backend::OpCategory::Sync).count();
    println!("\nTrace summary:");
    println!("  Total events: {}", events.len());
    println!("  Fusion events: {}", n_fusion);
    println!("  Sync points: {}", n_sync);
    println!("  Ground truth entries: {}", gt.len());

    // Write trace
    let trace_path = "validation_trace.js";
    write_trace(&events, trace_path).unwrap();
    println!("\nTrace written to: {}", trace_path);

    // Write ground truth
    let gt_json = json!({
        "description": "Ground truth for validation example — shapes and full data for every output tensor",
        "entries": gt,
    });
    let gt_path = "validation_ground_truth.json";
    std::fs::write(gt_path, serde_json::to_string_pretty(&gt_json).unwrap()).unwrap();
    println!("Ground truth written to: {}", gt_path);
    println!("\nRun:  python validate_trace.py");
}
