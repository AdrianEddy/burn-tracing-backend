use burn_backend::{
    Backend, Distribution, ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_backend::{BoolDType, FloatDType, IntDType, Shape, Slice};
use std::future::Future;
use std::ops::Range;

use crate::backend::Profiler;
use crate::trace::{OpCategory, trace_op, tensor_bytes, tensor_shape, record_int_creation};

const C: OpCategory = OpCategory::Int;

impl<B: Backend> IntTensorOps<Self> for Profiler<B> {
    fn int_from_data(data: TensorData, device: &Device<B>) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_from_data(data, device);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_from_data", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn int_into_data(tensor: IntTensor<B>) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let start = std::time::Instant::now();
        let fut = B::int_into_data(tensor);
        async move {
            let result = fut.await;
            let (shape, preview) = result.as_ref()
                .map(|data| crate::trace::extract_tensor_data_preview(data))
                .unwrap_or_default();
            crate::trace::record_op_sync("int_into_data", C, start, start.elapsed(), Some(shape), preview);
            result
        }
    }

    fn int_device(tensor: &IntTensor<B>) -> Device<B> {
        B::int_device(tensor)
    }

    fn int_to_device(tensor: IntTensor<B>, device: &Device<B>) -> IntTensor<B> {
        trace_with_shape!(C, "int_to_device", B::int_to_device(tensor, device), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_empty(shape: Shape, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_empty(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_empty", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn int_reshape(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B> {
        let in_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&tensor).iter().copied().collect();
        trace_with_shape!(C, "int_reshape", [in_shape], B::int_reshape(tensor, shape), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_slice(tensor: IntTensor<B>, slices: &[Slice]) -> IntTensor<B> {
        trace_with_shape!(C, "int_slice", B::int_slice(tensor, slices), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_slice_assign(tensor: IntTensor<B>, slices: &[Slice], value: IntTensor<B>) -> IntTensor<B> {
        trace_with_shape!(C, "int_slice_assign", B::int_slice_assign(tensor, slices, value), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_into_float(tensor: IntTensor<B>, out_dtype: FloatDType) -> FloatTensor<B> {
        trace_with_shape!(C, "int_into_float", B::int_into_float(tensor, out_dtype), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn int_mask_where(tensor: IntTensor<B>, mask: BoolTensor<B>, value: IntTensor<B>) -> IntTensor<B> {
        trace_with_shape!(C, "int_mask_where", B::int_mask_where(tensor, mask, value), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_mask_fill(tensor: IntTensor<B>, mask: BoolTensor<B>, value: Scalar) -> IntTensor<B> {
        trace_with_shape!(C, "int_mask_fill", B::int_mask_fill(tensor, mask, value), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_gather(dim: usize, tensor: IntTensor<B>, indices: IntTensor<B>) -> IntTensor<B> {
        trace_with_shape!(C, "int_gather", B::int_gather(dim, tensor, indices), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_scatter_add(dim: usize, tensor: IntTensor<B>, indices: IntTensor<B>, value: IntTensor<B>) -> IntTensor<B> {
        trace_with_shape!(C, "int_scatter_add", B::int_scatter_add(dim, tensor, indices, value), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_select(tensor: IntTensor<B>, dim: usize, indices: IntTensor<B>) -> IntTensor<B> {
        trace_with_shape!(C, "int_select", B::int_select(tensor, dim, indices), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_select_add(tensor: IntTensor<B>, dim: usize, indices: IntTensor<B>, value: IntTensor<B>) -> IntTensor<B> {
        trace_with_shape!(C, "int_select_add", B::int_select_add(tensor, dim, indices, value), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_repeat_dim(tensor: IntTensor<B>, dim: usize, times: usize) -> IntTensor<B> {
        trace_with_shape!(C, "int_repeat_dim", B::int_repeat_dim(tensor, dim, times), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_cat(tensors: Vec<IntTensor<B>>, dim: usize) -> IntTensor<B> {
        trace_with_shape!(C, "int_cat", B::int_cat(tensors, dim), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_equal(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_equal", B::int_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_not_equal(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_not_equal", B::int_not_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_equal_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_equal_elem", B::int_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_not_equal_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_not_equal_elem", B::int_not_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_greater(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_greater", B::int_greater(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_greater_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_greater_elem", B::int_greater_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_greater_equal(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_greater_equal", B::int_greater_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_greater_equal_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_greater_equal_elem", B::int_greater_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_lower(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_lower", B::int_lower(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_lower_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_lower_elem", B::int_lower_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_lower_equal(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_lower_equal", B::int_lower_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_lower_equal_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_lower_equal_elem", B::int_lower_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    delegate!(C, fn int_add(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_add_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn int_sub(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_sub_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn int_mul(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_mul_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn int_div(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_div_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn int_remainder(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_remainder_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn int_matmul(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_neg(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_powi(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_powi_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn int_powi_scalar_impl(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);

    fn int_zeros(shape: Shape, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_zeros(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_zeros", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn int_ones(shape: Shape, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_ones(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_ones", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn int_full(shape: Shape, fill_value: Scalar, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_full(shape, fill_value, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_full", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    delegate!(C, fn int_sum(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_sum_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_prod(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_prod_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_mean(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_mean_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_cumsum(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_cumprod(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_cummin(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_cummax(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_argmax(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_argmin(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_max(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_max_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);

    fn int_max_dim_with_indices(tensor: IntTensor<B>, dim: usize) -> (IntTensor<B>, IntTensor<B>) {
        let _g = trace_op("int_max_dim_with_indices", C);
        B::int_max_dim_with_indices(tensor, dim)
    }

    delegate!(C, fn int_max_abs(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_max_abs_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);
    delegate!(C, fn int_min(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_min_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>);

    fn int_min_dim_with_indices(tensor: IntTensor<B>, dim: usize) -> (IntTensor<B>, IntTensor<B>) {
        let _g = trace_op("int_min_dim_with_indices", C);
        B::int_min_dim_with_indices(tensor, dim)
    }

    delegate!(C, fn int_abs(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn int_transpose(tensor: IntTensor<B>) -> IntTensor<B>);

    fn int_swap_dims(tensor: IntTensor<B>, dim1: usize, dim2: usize) -> IntTensor<B> {
        trace_with_shape!(C, "int_swap_dims", B::int_swap_dims(tensor, dim1, dim2), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_permute(tensor: IntTensor<B>, axes: &[usize]) -> IntTensor<B> {
        trace_with_shape!(C, "int_permute", B::int_permute(tensor, axes), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_flip(tensor: IntTensor<B>, axes: &[usize]) -> IntTensor<B> {
        trace_with_shape!(C, "int_flip", B::int_flip(tensor, axes), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_random(shape: Shape, distribution: Distribution, device: &Device<B>) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_random(shape, distribution, device);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_random", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn int_arange(range: Range<i64>, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_arange(range, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_arange", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn int_arange_step(range: Range<i64>, step: usize, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        let start = std::time::Instant::now();
        let result = B::int_arange_step(range, step, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_int_creation::<B>("int_arange_step", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn int_clamp(tensor: IntTensor<B>, min: Scalar, max: Scalar) -> IntTensor<B> {
        trace_with_shape!(C, "int_clamp", B::int_clamp(tensor, min, max), data: crate::trace::maybe_capture_int_data::<B>)
    }

    delegate!(C, fn int_clamp_min(tensor: IntTensor<B>, min: Scalar) -> IntTensor<B>);
    delegate!(C, fn int_clamp_max(tensor: IntTensor<B>, max: Scalar) -> IntTensor<B>);

    fn int_any(tensor: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_any", B::int_any(tensor, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_any_dim(tensor: IntTensor<B>, dim: usize, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_any_dim", B::int_any_dim(tensor, dim, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_all(tensor: IntTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_all", B::int_all(tensor, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn int_all_dim(tensor: IntTensor<B>, dim: usize, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "int_all_dim", B::int_all_dim(tensor, dim, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    delegate!(C, fn int_sign(tensor: IntTensor<B>) -> IntTensor<B>);

    fn int_expand(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B> {
        trace_with_shape!(C, "int_expand", B::int_expand(tensor, shape), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_sort(tensor: IntTensor<B>, dim: usize, descending: bool) -> IntTensor<B> {
        trace_with_shape!(C, "int_sort", B::int_sort(tensor, dim, descending), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_sort_with_indices(tensor: IntTensor<B>, dim: usize, descending: bool) -> (IntTensor<B>, IntTensor<B>) {
        let _g = trace_op("int_sort_with_indices", C);
        B::int_sort_with_indices(tensor, dim, descending)
    }

    fn int_argsort(tensor: IntTensor<B>, dim: usize, descending: bool) -> IntTensor<B> {
        trace_with_shape!(C, "int_argsort", B::int_argsort(tensor, dim, descending), data: crate::trace::maybe_capture_int_data::<B>)
    }

    delegate!(C, fn bitwise_and(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn bitwise_and_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn bitwise_or(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn bitwise_or_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn bitwise_xor(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn bitwise_xor_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn bitwise_not(tensor: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn bitwise_left_shift(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn bitwise_left_shift_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);
    delegate!(C, fn bitwise_right_shift(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>);
    delegate!(C, fn bitwise_right_shift_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B>);

    fn int_cast(tensor: IntTensor<B>, dtype: IntDType) -> IntTensor<B> {
        trace_with_shape!(C, "int_cast", B::int_cast(tensor, dtype), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn int_unfold(tensor: IntTensor<B>, dim: usize, size: usize, step: usize) -> IntTensor<B> {
        trace_with_shape!(C, "int_unfold", B::int_unfold(tensor, dim, size, step), data: crate::trace::maybe_capture_int_data::<B>)
    }
}
