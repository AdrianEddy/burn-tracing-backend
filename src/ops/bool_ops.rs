use burn_backend::{
    Backend, ExecutionError, Scalar, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_backend::{BoolDType, FloatDType, IntDType, Shape, Slice};
use std::future::Future;

use crate::backend::Profiler;
use crate::trace::{OpCategory, tensor_bytes, tensor_shape, record_bool_creation};

const C: OpCategory = OpCategory::Bool;

impl<B: Backend> BoolTensorOps<Self> for Profiler<B> {
    fn bool_from_data(data: TensorData, device: &Device<B>) -> BoolTensor<B> {
        let start = std::time::Instant::now();
        let result = B::bool_from_data(data, device);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_bool_creation::<B>("bool_from_data", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn bool_into_data(tensor: BoolTensor<B>) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let start = std::time::Instant::now();
        let fut = B::bool_into_data(tensor);
        async move {
            let result = fut.await;
            let (shape, preview) = result.as_ref()
                .map(|data| crate::trace::extract_tensor_data_preview(data))
                .unwrap_or_default();
            crate::trace::record_op_sync("bool_into_data", C, start, start.elapsed(), Some(shape), preview);
            result
        }
    }

    fn bool_device(tensor: &BoolTensor<B>) -> Device<B> {
        B::bool_device(tensor)
    }

    fn bool_to_device(tensor: BoolTensor<B>, device: &Device<B>) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_to_device", B::bool_to_device(tensor, device), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_empty(shape: Shape, device: &Device<B>, dtype: BoolDType) -> BoolTensor<B> {
        let start = std::time::Instant::now();
        let result = B::bool_empty(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_bool_creation::<B>("bool_empty", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn bool_zeros(shape: Shape, device: &Device<B>, dtype: BoolDType) -> BoolTensor<B> {
        let start = std::time::Instant::now();
        let result = B::bool_zeros(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_bool_creation::<B>("bool_zeros", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn bool_ones(shape: Shape, device: &Device<B>, dtype: BoolDType) -> BoolTensor<B> {
        let start = std::time::Instant::now();
        let result = B::bool_ones(shape, device, dtype);
        let bytes = tensor_bytes(&result);
        let shape = tensor_shape(&result);
        record_bool_creation::<B>("bool_ones", C, start, start.elapsed(), bytes, shape, &result);
        result
    }

    fn bool_into_int(tensor: BoolTensor<B>, out_dtype: IntDType) -> IntTensor<B> {
        trace_with_shape!(C, "bool_into_int", B::bool_into_int(tensor, out_dtype), data: crate::trace::maybe_capture_int_data::<B>)
    }

    fn bool_into_float(tensor: BoolTensor<B>, out_dtype: FloatDType) -> FloatTensor<B> {
        trace_with_shape!(C, "bool_into_float", B::bool_into_float(tensor, out_dtype), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn bool_reshape(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B> {
        let in_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&tensor).iter().copied().collect();
        trace_with_shape!(C, "bool_reshape", [in_shape], B::bool_reshape(tensor, shape), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_slice(tensor: BoolTensor<B>, slices: &[Slice]) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_slice", B::bool_slice(tensor, slices), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_slice_assign(tensor: BoolTensor<B>, slices: &[Slice], value: BoolTensor<B>) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_slice_assign", B::bool_slice_assign(tensor, slices, value), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_mask_where(tensor: BoolTensor<B>, mask: BoolTensor<B>, value: BoolTensor<B>) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_mask_where", B::bool_mask_where(tensor, mask, value), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_mask_fill(tensor: BoolTensor<B>, mask: BoolTensor<B>, value: Scalar) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_mask_fill", B::bool_mask_fill(tensor, mask, value), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_gather(dim: usize, tensor: BoolTensor<B>, indices: IntTensor<B>) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_gather", B::bool_gather(dim, tensor, indices), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_scatter_or(dim: usize, tensor: BoolTensor<B>, indices: IntTensor<B>, value: BoolTensor<B>) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_scatter_or", B::bool_scatter_or(dim, tensor, indices, value), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_select(tensor: BoolTensor<B>, dim: usize, indices: IntTensor<B>) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_select", B::bool_select(tensor, dim, indices), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_select_or(tensor: BoolTensor<B>, dim: usize, indices: IntTensor<B>, value: BoolTensor<B>) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_select_or", B::bool_select_or(tensor, dim, indices, value), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_repeat_dim(tensor: BoolTensor<B>, dim: usize, times: usize) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_repeat_dim", B::bool_repeat_dim(tensor, dim, times), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_cat(tensors: Vec<BoolTensor<B>>, dim: usize) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_cat", B::bool_cat(tensors, dim), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    delegate!(C, fn bool_equal(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_not_equal(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_equal_elem(lhs: BoolTensor<B>, rhs: Scalar) -> BoolTensor<B>);
    delegate!(C, fn bool_not_equal_elem(lhs: BoolTensor<B>, rhs: Scalar) -> BoolTensor<B>);
    delegate!(C, fn bool_not(tensor: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_and(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_or(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_xor(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_transpose(tensor: BoolTensor<B>) -> BoolTensor<B>);

    fn bool_swap_dims(tensor: BoolTensor<B>, dim1: usize, dim2: usize) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_swap_dims", B::bool_swap_dims(tensor, dim1, dim2), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_permute(tensor: BoolTensor<B>, axes: &[usize]) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_permute", B::bool_permute(tensor, axes), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_flip(tensor: BoolTensor<B>, axes: &[usize]) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_flip", B::bool_flip(tensor, axes), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    delegate!(C, fn bool_any(tensor: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_any_dim(tensor: BoolTensor<B>, dim: usize) -> BoolTensor<B>);
    delegate!(C, fn bool_all(tensor: BoolTensor<B>) -> BoolTensor<B>);
    delegate!(C, fn bool_all_dim(tensor: BoolTensor<B>, dim: usize) -> BoolTensor<B>);

    fn bool_argwhere(tensor: BoolTensor<B>, out_dtype: IntDType) -> impl Future<Output = IntTensor<B>> + 'static + Send {
        let start = std::time::Instant::now();
        let fut = B::bool_argwhere(tensor, out_dtype);
        async move {
            let result = fut.await;
            crate::trace::record_op("bool_argwhere", C, start, start.elapsed());
            result
        }
    }

    fn bool_expand(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_expand", B::bool_expand(tensor, shape), data: crate::trace::maybe_capture_bool_data::<B>)
    }

    fn bool_unfold(tensor: BoolTensor<B>, dim: usize, size: usize, step: usize) -> BoolTensor<B> {
        trace_with_shape!(C, "bool_unfold", B::bool_unfold(tensor, dim, size, step), data: crate::trace::maybe_capture_bool_data::<B>)
    }
}
