use burn_backend::{Backend, DType, DTypeUsageSet, ExecutionError, TensorData};
use std::marker::PhantomData;

use crate::trace::{OpCategory, trace_op};

#[cfg(feature = "fusion")]
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
#[cfg(feature = "fusion")]
use burn_ir::{BackendIr, TensorHandle};

/// A backend decorator that traces all tensor operations with CPU timing.
///
/// Wrap any Burn backend with `Profiler<B>` to record every operation,
/// detect sync points, and export HTML visualizations.
///
/// # Example
/// ```ignore
/// type MyBackend = Profiler<NdArray>;
/// ```
#[derive(Clone, Debug, Default)]
pub struct Profiler<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> Backend for Profiler<B> {
    type Device = B::Device;
    type FloatTensorPrimitive = B::FloatTensorPrimitive;
    type FloatElem = B::FloatElem;
    type IntTensorPrimitive = B::IntTensorPrimitive;
    type IntElem = B::IntElem;
    type BoolTensorPrimitive = B::BoolTensorPrimitive;
    type BoolElem = B::BoolElem;
    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;

    fn name(device: &Self::Device) -> String {
        format!("profiler<{}>", B::name(device))
    }

    fn seed(device: &Self::Device, seed: u64) {
        B::seed(device, seed);
    }

    fn ad_enabled(device: &Self::Device) -> bool {
        B::ad_enabled(device)
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let _g = trace_op("sync", OpCategory::Sync);
        B::sync(device)
    }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        B::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        B::memory_cleanup(device);
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        B::staging(data, device);
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        B::supports_dtype(device, dtype)
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }
}

// ---------------------------------------------------------------------------
// BackendIr – delegate to the inner backend (fusion feature only)
// ---------------------------------------------------------------------------

#[cfg(feature = "fusion")]
impl<B: Backend + BackendIr> BackendIr for Profiler<B> {
    type Handle = B::Handle;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        B::float_tensor(handle)
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        B::int_tensor(handle)
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        B::bool_tensor(handle)
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        B::quantized_tensor(handle)
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        B::float_tensor_handle(tensor)
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        B::int_tensor_handle(tensor)
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        B::bool_tensor_handle(tensor)
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        B::quantized_tensor_handle(tensor)
    }
}

// ---------------------------------------------------------------------------
// FusionBackend – connect Profiler<CubeBackend<R>> to ProfiledFusionRuntime<R>
// ---------------------------------------------------------------------------

#[cfg(feature = "fusion")]
impl<R, F, I, BT> burn_fusion::FusionBackend for Profiler<burn_cubecl::CubeBackend<R, F, I, BT>>
where
    R: burn_cubecl::CubeRuntime,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
{
    type FusionRuntime = crate::fusion_runtime::ProfiledFusionRuntime<R>;
    type FullPrecisionBackend = Profiler<burn_cubecl::CubeBackend<R, f32, i32, BT>>;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        <burn_cubecl::CubeBackend<R, F, I, BT> as burn_fusion::FusionBackend>::cast_float(tensor, dtype)
    }
}
