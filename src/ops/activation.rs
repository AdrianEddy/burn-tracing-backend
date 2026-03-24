use burn_backend::{
    Backend, Scalar,
    ops::ActivationOps,
    tensor::FloatTensor,
};

use crate::backend::Profiler;
use crate::trace::OpCategory;

const C: OpCategory = OpCategory::Activation;

impl<B: Backend> ActivationOps<Self> for Profiler<B> {
    delegate!(C, fn relu(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn leaky_relu(tensor: FloatTensor<B>, negative_slope: Scalar) -> FloatTensor<B>);
    delegate!(C, fn relu_backward(output: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn gelu(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn prelu(tensor: FloatTensor<B>, alpha: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn gelu_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn sigmoid(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn sigmoid_backward(output: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B>);

    fn hard_sigmoid(tensor: FloatTensor<B>, alpha: Scalar, beta: Scalar) -> FloatTensor<B> {
        trace_with_shape!(C, "hard_sigmoid", B::hard_sigmoid(tensor, alpha, beta), data: crate::trace::maybe_capture_float_data::<B>)
    }

    delegate!(C, fn log_sigmoid(tensor: FloatTensor<B>) -> FloatTensor<B>);
    delegate!(C, fn log_sigmoid_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B>);
}
