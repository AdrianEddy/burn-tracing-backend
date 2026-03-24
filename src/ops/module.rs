use burn_backend::{
    Backend,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
        DeformConvOptions, InterpolateOptions, MaxPool1dBackward, MaxPool1dWithIndices,
        MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps, UnfoldOptions,
    },
    tensor::{BoolTensor, FloatTensor, IntTensor},
};

use crate::backend::Profiler;
use crate::trace::{OpCategory, trace_op};

const C: OpCategory = OpCategory::Module;

impl<B: Backend> ModuleOps<Self> for Profiler<B> {
    delegate!(C, fn embedding(weights: FloatTensor<B>, indices: IntTensor<B>) -> FloatTensor<B>);

    fn embedding_backward(
        weights: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "embedding_backward", B::embedding_backward(weights, output_grad, indices), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv1d", B::conv1d(x, weight, bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv1d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv1d_x_backward", B::conv1d_x_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv1d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv1d_weight_backward", B::conv1d_weight_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv1d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv1d_bias_backward", B::conv1d_bias_backward(x, bias, output_grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv2d", B::conv2d(x, weight, bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv2d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv2d_x_backward", B::conv2d_x_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv2d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv2d_weight_backward", B::conv2d_weight_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv2d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv2d_bias_backward", B::conv2d_bias_backward(x, bias, output_grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn deform_conv2d(
        x: FloatTensor<B>,
        offset: FloatTensor<B>,
        weight: FloatTensor<B>,
        mask: Option<FloatTensor<B>>,
        bias: Option<FloatTensor<B>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "deform_conv2d", B::deform_conv2d(x, offset, weight, mask, bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<B>,
        offset: FloatTensor<B>,
        weight: FloatTensor<B>,
        mask: Option<FloatTensor<B>>,
        bias: Option<FloatTensor<B>>,
        output_grad: FloatTensor<B>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let _g = trace_op("deform_conv2d_backward", C);
        let r = B::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options);
        DeformConv2dBackward::new(r.x_grad, r.offset_grad, r.weight_grad, r.mask_grad, r.bias_grad)
    }

    fn conv3d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv3d", B::conv3d(x, weight, bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv3d_x_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv3d_x_backward", B::conv3d_x_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv3d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv3d_weight_backward", B::conv3d_weight_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv3d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv3d_bias_backward", B::conv3d_bias_backward(x, bias, output_grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose1d", B::conv_transpose1d(x, weight, bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose1d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose1d_x_backward", B::conv_transpose1d_x_backward(weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose1d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose1d_weight_backward", B::conv_transpose1d_weight_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose1d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose1d_bias_backward", B::conv_transpose1d_bias_backward(x, bias, output_grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose2d", B::conv_transpose2d(x, weight, bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose2d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose2d_x_backward", B::conv_transpose2d_x_backward(weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose2d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose2d_weight_backward", B::conv_transpose2d_weight_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose2d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose2d_bias_backward", B::conv_transpose2d_bias_backward(x, bias, output_grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose3d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose3d", B::conv_transpose3d(x, weight, bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose3d_x_backward(
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose3d_x_backward", B::conv_transpose3d_x_backward(weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose3d_weight_backward(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose3d_weight_backward", B::conv_transpose3d_weight_backward(x, weight, output_grad, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn conv_transpose3d_bias_backward(
        x: FloatTensor<B>,
        bias: FloatTensor<B>,
        output_grad: FloatTensor<B>,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "conv_transpose3d_bias_backward", B::conv_transpose3d_bias_backward(x, bias, output_grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn unfold4d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        options: UnfoldOptions,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "unfold4d", B::unfold4d(x, kernel_size, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn avg_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "avg_pool1d", B::avg_pool1d(x, kernel_size, stride, padding, count_include_pad, ceil_mode), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn avg_pool1d_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "avg_pool1d_backward", B::avg_pool1d_backward(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn avg_pool2d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "avg_pool2d", B::avg_pool2d(x, kernel_size, stride, padding, count_include_pad, ceil_mode), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "avg_pool2d_backward", B::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn adaptive_avg_pool2d(x: FloatTensor<B>, output_size: [usize; 2]) -> FloatTensor<B> {
        trace_with_shape!(C, "adaptive_avg_pool2d", B::adaptive_avg_pool2d(x, output_size), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn adaptive_avg_pool2d_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "adaptive_avg_pool2d_backward", B::adaptive_avg_pool2d_backward(x, grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn adaptive_avg_pool1d(x: FloatTensor<B>, output_size: usize) -> FloatTensor<B> {
        trace_with_shape!(C, "adaptive_avg_pool1d", B::adaptive_avg_pool1d(x, output_size), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn adaptive_avg_pool1d_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        trace_with_shape!(C, "adaptive_avg_pool1d_backward", B::adaptive_avg_pool1d_backward(x, grad), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn max_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "max_pool1d", B::max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> MaxPool1dWithIndices<Self> {
        let _g = trace_op("max_pool1d_with_indices", C);
        let r = B::max_pool1d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
        MaxPool1dWithIndices::new(r.output, r.indices)
    }

    #[allow(clippy::too_many_arguments)]
    fn max_pool1d_with_indices_backward(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> MaxPool1dBackward<Self> {
        let _g = trace_op("max_pool1d_with_indices_backward", C);
        let r = B::max_pool1d_with_indices_backward(x, kernel_size, stride, padding, dilation, ceil_mode, output_grad, indices);
        MaxPool1dBackward::new(r.x_grad)
    }

    fn max_pool2d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "max_pool2d", B::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let _g = trace_op("max_pool2d_with_indices", C);
        let r = B::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
        MaxPool2dWithIndices::new(r.output, r.indices)
    }

    #[allow(clippy::too_many_arguments)]
    fn max_pool2d_with_indices_backward(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> MaxPool2dBackward<Self> {
        let _g = trace_op("max_pool2d_with_indices_backward", C);
        let r = B::max_pool2d_with_indices_backward(x, kernel_size, stride, padding, dilation, ceil_mode, output_grad, indices);
        MaxPool2dBackward::new(r.x_grad)
    }

    fn interpolate(
        x: FloatTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "interpolate", B::interpolate(x, output_size, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn interpolate_backward(
        x: FloatTensor<B>,
        grad: FloatTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "interpolate_backward", B::interpolate_backward(x, grad, output_size, options), data: crate::trace::maybe_capture_float_data::<B>)
    }

    fn attention(
        query: FloatTensor<B>,
        key: FloatTensor<B>,
        value: FloatTensor<B>,
        mask: Option<BoolTensor<B>>,
        attn_bias: Option<FloatTensor<B>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<B> {
        trace_with_shape!(C, "attention", B::attention(query, key, value, mask, attn_bias, options), data: crate::trace::maybe_capture_float_data::<B>)
    }
}
