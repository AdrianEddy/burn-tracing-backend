use burn_backend::{
    Backend, ExecutionError, TensorData,
    ops::QTensorOps,
    tensor::{
        Device, FloatTensor, QuantizedTensor,
        quantization::QuantizationParametersPrimitive,
    },
    quantization::QuantScheme,
};
use burn_backend::{Shape, Slice};
use std::future::Future;

use crate::backend::Profiler;
use crate::trace::{OpCategory, trace_op};

const C: OpCategory = OpCategory::Quantized;

impl<B: Backend> QTensorOps<Self> for Profiler<B> {
    fn q_from_data(data: TensorData, device: &Device<B>) -> QuantizedTensor<B> {
        let _g = trace_op("q_from_data", C);
        B::q_from_data(data, device)
    }

    fn quantize(
        tensor: FloatTensor<B>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<B> {
        let _g = trace_op("quantize", C);
        B::quantize(tensor, scheme, QuantizationParametersPrimitive {
            scales: qparams.scales,
        })
    }

    fn quantize_dynamic(tensor: FloatTensor<B>, scheme: &QuantScheme) -> QuantizedTensor<B> {
        let _g = trace_op("quantize_dynamic", C);
        B::quantize_dynamic(tensor, scheme)
    }

    fn dequantize(tensor: QuantizedTensor<B>) -> FloatTensor<B> {
        let _g = trace_op("dequantize", C);
        B::dequantize(tensor)
    }

    fn q_device(tensor: &QuantizedTensor<B>) -> Device<B> {
        B::q_device(tensor)
    }

    fn q_to_device(tensor: QuantizedTensor<B>, device: &Device<B>) -> QuantizedTensor<B> {
        let _g = trace_op("q_to_device", C);
        B::q_to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<B>, shape: Shape) -> QuantizedTensor<B> {
        let in_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&tensor).iter().copied().collect();
        trace_with_shape!(C, "q_reshape", [in_shape], B::q_reshape(tensor, shape), data: crate::trace::maybe_capture_quantized_data::<B>)
    }

    fn q_into_data(tensor: QuantizedTensor<B>) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let start = std::time::Instant::now();
        let fut = B::q_into_data(tensor);
        async move {
            let result = fut.await;
            let (shape, preview) = result.as_ref()
                .map(|data| crate::trace::extract_tensor_data_preview(data))
                .unwrap_or_default();
            crate::trace::record_op_sync("q_into_data", C, start, start.elapsed(), Some(shape), preview);
            result
        }
    }

    fn q_expand(tensor: QuantizedTensor<B>, shape: Shape) -> QuantizedTensor<B> {
        let _g = trace_op("q_expand", C);
        B::q_expand(tensor, shape)
    }

    fn q_swap_dims(tensor: QuantizedTensor<B>, dim1: usize, dim2: usize) -> QuantizedTensor<B> {
        let _g = trace_op("q_swap_dims", C);
        B::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<B>, axes: &[usize]) -> QuantizedTensor<B> {
        let _g = trace_op("q_permute", C);
        B::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<B>, axes: &[usize]) -> QuantizedTensor<B> {
        let _g = trace_op("q_flip", C);
        B::q_flip(tensor, axes)
    }

    fn q_select(tensor: QuantizedTensor<B>, dim: usize, indices: burn_backend::tensor::IntTensor<B>) -> QuantizedTensor<B> {
        let _g = trace_op("q_select", C);
        B::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<B>, slices: &[Slice]) -> QuantizedTensor<B> {
        let _g = trace_op("q_slice", C);
        B::q_slice(tensor, slices)
    }

    // All arithmetic/math operations have default implementations that dequantize -> operate -> requantize.
    // These defaults call through our traced dequantize/quantize/float_* methods, so we get tracing
    // of the decomposed ops automatically.
}
