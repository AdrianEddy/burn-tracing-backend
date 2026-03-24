use burn_backend::{
    Backend, Distribution, ExecutionError, Scalar, TensorData,
    ops::{FloatTensorOps, GridSampleOptions},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_backend::{BoolDType, FloatDType, IntDType, Shape, Slice};
use std::future::Future;

use crate::backend::Profiler;
use crate::trace::{OpCategory, trace_op, tensor_bytes, tensor_shape, record_float_creation};

const C: OpCategory = OpCategory::Float;

impl<B: Backend> FloatTensorOps<Self> for Profiler<B> {
    fn float_from_data(data: TensorData, device: &Device<B>) -> FloatTensor<B> {
        let start = std::time::Instant::now();
        let result = B::float_from_data(data, device);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_float_creation::<B>("float_from_data", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn float_random(shape: Shape, distribution: Distribution, device: &Device<B>) -> FloatTensor<B> {
        let start = std::time::Instant::now();
        let result = B::float_random(shape, distribution, device);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_float_creation::<B>("float_random", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn float_zeros(shape: Shape, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B> {
        let start = std::time::Instant::now();
        let result = B::float_zeros(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_float_creation::<B>("float_zeros", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn float_ones(shape: Shape, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B> {
        let start = std::time::Instant::now();
        let result = B::float_ones(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_float_creation::<B>("float_ones", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn float_full(shape: Shape, fill_value: Scalar, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B> {
        let start = std::time::Instant::now();
        let result = B::float_full(shape, fill_value, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_float_creation::<B>("float_full", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn float_into_data(tensor: FloatTensor<B>) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let start = std::time::Instant::now();
        let fut = B::float_into_data(tensor);
        async move {
            let result = fut.await;
            let (shape, preview) = result.as_ref()
                .map(|data| crate::trace::extract_tensor_data_preview(data))
                .unwrap_or_default();
            crate::trace::record_op_sync("float_into_data", C, start, start.elapsed(), Some(shape), preview);
            result
        }
    }

    fn float_device(tensor: &FloatTensor<B>) -> Device<B> {
        B::float_device(tensor)
    }

    fn float_to_device(tensor: FloatTensor<B>, device: &Device<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "float_to_device", B::float_to_device(tensor, device), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_into_int(tensor: FloatTensor<B>, out_dtype: IntDType) -> IntTensor<B> {
        trace_with_shape!(C, "float_into_int", B::float_into_int(tensor, out_dtype), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn float_empty(shape: Shape, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B> {
        let start = std::time::Instant::now();
        let result = B::float_empty(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_float_creation::<B>("float_empty", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    delegate!(C, fn float_add(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_add_scalar(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_sub(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_sub_scalar(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_mul(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_mul_scalar(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_div(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_div_scalar(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_remainder(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_remainder_scalar(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_matmul(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_cross(lhs: FloatTensor<B>, rhs: FloatTensor<B>, dim: usize) -> FloatTensor<B>);
    delegate!(C, fn float_neg(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_recip(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_transpose(tensor: FloatTensor<B>) -> FloatTensor<B>);

    fn float_swap_dims(tensor: FloatTensor<B>, dim1: usize, dim2: usize) -> FloatTensor<B> {
        trace_with_shape!(C, "float_swap_dims", B::float_swap_dims(tensor, dim1, dim2), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_permute(tensor: FloatTensor<B>, axes: &[usize]) -> FloatTensor<B> {
        trace_with_shape!(C, "float_permute", B::float_permute(tensor, axes), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_flip(tensor: FloatTensor<B>, axes: &[usize]) -> FloatTensor<B> {
        trace_with_shape!(C, "float_flip", B::float_flip(tensor, axes), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_reshape(tensor: FloatTensor<B>, shape: Shape) -> FloatTensor<B> {
        let in_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&tensor).iter().copied().collect();
        trace_with_shape!(C, "float_reshape", [in_shape], B::float_reshape(tensor, shape), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_gather(dim: usize, tensor: FloatTensor<B>, indices: IntTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "float_gather", B::float_gather(dim, tensor, indices), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_scatter_add(dim: usize, tensor: FloatTensor<B>, indices: IntTensor<B>, value: FloatTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "float_scatter_add", B::float_scatter_add(dim, tensor, indices, value), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_select(tensor: FloatTensor<B>, dim: usize, indices: IntTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "float_select", B::float_select(tensor, dim, indices), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_select_add(tensor: FloatTensor<B>, dim: usize, indices: IntTensor<B>, value: FloatTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "float_select_add", B::float_select_add(tensor, dim, indices, value), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_slice(tensor: FloatTensor<B>, slices: &[Slice]) -> FloatTensor<B> {
        trace_with_shape!(C, "float_slice", B::float_slice(tensor, slices), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_slice_assign(tensor: FloatTensor<B>, slices: &[Slice], value: FloatTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "float_slice_assign", B::float_slice_assign(tensor, slices, value), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_mask_where(tensor: FloatTensor<B>, mask: BoolTensor<B>, value: FloatTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "float_mask_where", B::float_mask_where(tensor, mask, value), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_mask_fill(tensor: FloatTensor<B>, mask: BoolTensor<B>, value: Scalar) -> FloatTensor<B> {
        trace_with_shape!(C, "float_mask_fill", B::float_mask_fill(tensor, mask, value), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_equal", B::float_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_not_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_not_equal", B::float_not_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_equal_elem(lhs: FloatTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_equal_elem", B::float_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_not_equal_elem(lhs: FloatTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_not_equal_elem", B::float_not_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_greater(lhs: FloatTensor<B>, rhs: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_greater", B::float_greater(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_greater_elem(lhs: FloatTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_greater_elem", B::float_greater_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_greater_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_greater_equal", B::float_greater_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_greater_equal_elem", B::float_greater_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_lower(lhs: FloatTensor<B>, rhs: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_lower", B::float_lower(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_lower_elem(lhs: FloatTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_lower_elem", B::float_lower_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_lower_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_lower_equal", B::float_lower_equal(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_lower_equal_elem", B::float_lower_equal_elem(lhs, rhs, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_detach(tensor: FloatTensor<B>) -> FloatTensor<B> {
        B::float_detach(tensor)
    }

    fn float_set_require_grad(tensor: FloatTensor<B>, require_grad: bool) -> FloatTensor<B> {
        B::float_set_require_grad(tensor, require_grad)
    }

    fn float_is_require_grad(tensor: &FloatTensor<B>) -> bool {
        B::float_is_require_grad(tensor)
    }

    delegate!(C, fn float_sum(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_sum_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);
    delegate!(C, fn float_prod(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_prod_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);
    delegate!(C, fn float_mean(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_mean_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);
    delegate!(C, fn float_cumsum(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);
    delegate!(C, fn float_cumprod(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);
    delegate!(C, fn float_cummin(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);
    delegate!(C, fn float_cummax(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);

    fn float_cast(tensor: FloatTensor<B>, dtype: FloatDType) -> FloatTensor<B> {
        trace_with_shape!(C, "float_cast", B::float_cast(tensor, dtype), data: crate::trace::maybe_capture_float_data::<B>)
    }

    delegate!(C, fn float_exp(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_log(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_log1p(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_powf(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_powi(lhs: FloatTensor<B>, rhs: IntTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_powi_scalar(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_powi_scalar_impl(lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_powf_scalar(tensor: FloatTensor<B>, value: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_powf_scalar_impl(tensor: FloatTensor<B>, value: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_sqrt(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_abs(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_cos(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_sin(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_tan(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_cosh(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_sinh(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_tanh(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_acos(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_acosh(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_asin(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_asinh(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_atan(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_atanh(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_atan2(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_round(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_floor(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_ceil(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_trunc(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_erf(tensor: FloatTensor<B>) -> FloatTensor<B>);

    fn float_cat(tensors: Vec<FloatTensor<B>>, dim: usize) -> FloatTensor<B> {
        trace_with_shape!(C, "float_cat", B::float_cat(tensors, dim), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_argmax(tensor: FloatTensor<B>, dim: usize, out_dtype: IntDType) -> IntTensor<B> {
        trace_with_shape!(C, "float_argmax", B::float_argmax(tensor, dim, out_dtype), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn float_argmin(tensor: FloatTensor<B>, dim: usize, out_dtype: IntDType) -> IntTensor<B> {
        trace_with_shape!(C, "float_argmin", B::float_argmin(tensor, dim, out_dtype), data: crate::trace::maybe_capture_int_data::<B>)
    }

    delegate!(C, fn float_max(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_max_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);

    fn float_max_dim_with_indices(tensor: FloatTensor<B>, dim: usize, indices_dtype: IntDType) -> (FloatTensor<B>, IntTensor<B>) {
        let _g = trace_op("float_max_dim_with_indices", C);
        B::float_max_dim_with_indices(tensor, dim, indices_dtype)
    }

    delegate!(C, fn float_min(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_min_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);

    fn float_min_dim_with_indices(tensor: FloatTensor<B>, dim: usize, indices_dtype: IntDType) -> (FloatTensor<B>, IntTensor<B>) {
        let _g = trace_op("float_min_dim_with_indices", C);
        B::float_min_dim_with_indices(tensor, dim, indices_dtype)
    }

    delegate!(C, fn float_max_abs(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn float_max_abs_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>);

    fn float_clamp(tensor: FloatTensor<B>, min: Scalar, max: Scalar) -> FloatTensor<B> {
        trace_with_shape!(C, "float_clamp", B::float_clamp(tensor, min, max), data: crate::trace::maybe_capture_float_data::<B>)
    }

    delegate!(C, fn float_clamp_min(tensor: FloatTensor<B>, min: Scalar) -> FloatTensor<B>);
    delegate!(C, fn float_clamp_max(tensor: FloatTensor<B>, max: Scalar) -> FloatTensor<B>);

    fn float_any(tensor: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_any", B::float_any(tensor, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_any_dim(tensor: FloatTensor<B>, dim: usize, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_any_dim", B::float_any_dim(tensor, dim, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_all(tensor: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_all", B::float_all(tensor, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_all_dim(tensor: FloatTensor<B>, dim: usize, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_all_dim", B::float_all_dim(tensor, dim, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    delegate!(C, fn float_sign(tensor: FloatTensor<B>) -> FloatTensor<B>);

    fn float_expand(tensor: FloatTensor<B>, shape: Shape) -> FloatTensor<B> {
        trace_with_shape!(C, "float_expand", B::float_expand(tensor, shape), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_sort(tensor: FloatTensor<B>, dim: usize, descending: bool) -> FloatTensor<B> {
        trace_with_shape!(C, "float_sort", B::float_sort(tensor, dim, descending), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_sort_with_indices(tensor: FloatTensor<B>, dim: usize, descending: bool, indices_dtype: IntDType) -> (FloatTensor<B>, IntTensor<B>) {
        let _g = trace_op("float_sort_with_indices", C);
        B::float_sort_with_indices(tensor, dim, descending, indices_dtype)
    }

    fn float_argsort(tensor: FloatTensor<B>, dim: usize, descending: bool, out_dtype: IntDType) -> IntTensor<B> {
        trace_with_shape!(C, "float_argsort", B::float_argsort(tensor, dim, descending, out_dtype), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn float_repeat_dim(tensor: FloatTensor<B>, dim: usize, times: usize) -> FloatTensor<B> {
        trace_with_shape!(C, "float_repeat_dim", B::float_repeat_dim(tensor, dim, times), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_unfold(tensor: FloatTensor<B>, dim: usize, size: usize, step: usize) -> FloatTensor<B> {
        trace_with_shape!(C, "float_unfold", B::float_unfold(tensor, dim, size, step), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn float_is_nan(tensor: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_is_nan", B::float_is_nan(tensor, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_is_inf(tensor: FloatTensor<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        trace_with_shape!(C, "float_is_inf", B::float_is_inf(tensor, out_dtype), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn float_grid_sample_2d(tensor: FloatTensor<B>, grid: FloatTensor<B>, options: GridSampleOptions) -> FloatTensor<B> {
        trace_with_shape!(C, "float_grid_sample_2d", B::float_grid_sample_2d(tensor, grid, options), data: crate::trace::maybe_capture_float_data::<B>)
    }
}
