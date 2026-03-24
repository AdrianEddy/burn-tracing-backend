//! Profiled fusion runtime that intercepts optimization execution.
//!
//! This module wraps `FusionCubeRuntime<R>` to capture concrete fusion events
//! (kind, number of fused ops, execution time, per-op names + shapes).

use std::sync::Arc;
use std::time::Instant;

use burn_cubecl::CubeRuntime;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl_fusion::{
    CubeFusionHandle, FallbackOperation,
    optim::{
        CubeOptimization, CubeOptimizationState,
        elemwise::ElemwiseOptimization,
        matmul::MatmulOptimization,
        reduce::ReduceOptimization,
        reduce_broadcasted::ReduceBroadcastedOptimization,
    },
};
use burn_fusion::{
    FusionRuntime, NumOperations, OperationFuser, Optimization,
    stream::{Context, Operation, OrderedExecution},
};
use burn_ir::{OperationIr, TensorId};
use core::marker::PhantomData;
use std::collections::{HashMap, HashSet};

use crate::trace::{FusedOpInfo, FusionKind, FusionTensorData, record_fusion_execute};

// ---------------------------------------------------------------------------
// OperationIr → FusedOpInfo extraction
// ---------------------------------------------------------------------------

/// Extract variant name from a Debug-formatted enum value.
/// E.g. "Add(BinaryOpIr { ... })" → "Add"
fn variant_name(debug: &str) -> &str {
    debug.split('(').next().unwrap_or(debug)
}

/// Extract a human-readable name + shapes from an OperationIr.
fn op_ir_info(op: &OperationIr) -> FusedOpInfo {
    let name = match op {
        OperationIr::BaseFloat(inner) => format!("float::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::BaseInt(inner) => format!("int::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::BaseBool(inner) => format!("bool::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::NumericFloat(_, inner) => format!("float::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::NumericInt(_, inner) => format!("int::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::Bool(inner) => format!("bool::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::Int(inner) => format!("int::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::Float(_, inner) => format!("float::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::Module(inner) => format!("module::{}", variant_name(&format!("{inner:?}"))),
        OperationIr::Init(_) => "Init".to_string(),
        OperationIr::Custom(inner) => format!("custom::{}", inner.id),
        OperationIr::Drop(_) => "Drop".to_string(),
    };

    let inputs: Vec<_> = op.inputs().collect();
    let outputs: Vec<_> = op.outputs().collect();

    let input_shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape.iter().copied().collect()).collect();
    let output_shapes: Vec<Vec<usize>> = outputs.iter().map(|t| t.shape.iter().copied().collect()).collect();
    let input_ids: Vec<String> = inputs.iter().map(|t| format!("{}", t.id)).collect();
    let output_ids: Vec<String> = outputs.iter().map(|t| format!("{}", t.id)).collect();

    FusedOpInfo {
        name,
        input_shapes,
        output_shapes,
        input_ids,
        output_ids,
        is_fallback: None,
    }
}

// ---------------------------------------------------------------------------
// ProfiledFusionRuntime
// ---------------------------------------------------------------------------

/// A fusion runtime that delegates to `FusionCubeRuntime<R>` while recording
/// every optimization execution as a [`TraceEvent`](crate::trace::TraceEvent).
#[derive(Debug)]
pub struct ProfiledFusionRuntime<R: CubeRuntime> {
    _phantom: PhantomData<R>,
}

impl<R: CubeRuntime> FusionRuntime for ProfiledFusionRuntime<R> {
    type OptimizationState = CubeOptimizationState;
    type Optimization = ProfiledOptimization<R>;
    type FusionHandle = CubeFusionHandle<R>;
    type FusionDevice = <FusionCubeRuntime<R> as FusionRuntime>::FusionDevice;

    fn fusers(
        device: Self::FusionDevice,
    ) -> Vec<Box<dyn OperationFuser<Self::Optimization>>> {
        let inner_fusers = FusionCubeRuntime::<R>::fusers(device);
        inner_fusers
            .into_iter()
            .map(|f| -> Box<dyn OperationFuser<Self::Optimization>> {
                Box::new(ProfiledFuser {
                    inner: f,
                    fused_op_info: Vec::new(),
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ProfiledOptimization
// ---------------------------------------------------------------------------

/// Wraps a `CubeOptimization<R>` and records timing + fusion kind on execute.
pub struct ProfiledOptimization<R: CubeRuntime> {
    pub(crate) inner: CubeOptimization<R>,
    /// Info about each op that was fused into this optimization.
    pub(crate) fused_op_info: Vec<FusedOpInfo>,
}

impl<R: CubeRuntime> core::fmt::Debug for ProfiledOptimization<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProfiledOptimization({:?})", self.inner)
    }
}

impl<R: CubeRuntime> NumOperations for ProfiledOptimization<R> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<R: CubeRuntime> Optimization<ProfiledFusionRuntime<R>> for ProfiledOptimization<R> {
    fn execute(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        execution: &OrderedExecution<ProfiledFusionRuntime<R>>,
    ) {
        let (kind, num_ops) = match &self.inner {
            CubeOptimization::ElementWise(op) => (FusionKind::ElementWise, op.num_ops_fused()),
            CubeOptimization::Matmul(op) => (FusionKind::Matmul, op.num_ops_fused()),
            CubeOptimization::Reduce(op) => (FusionKind::Reduce, op.num_ops_fused()),
            CubeOptimization::ReduceBroadcasted(op) => {
                (FusionKind::ReduceBroadcasted, op.num_ops_fused())
            }
        };

        // Clone the fused op info and resolve relative shape IDs to real dimensions.
        // OperationIr uses symbolic/relative shape IDs during fusion; the context
        // mapping translates them to actual dimension values.
        let mut fused_ops = self.fused_op_info.clone();
        let shape_map = context.shapes_relative2global;
        for op_info in fused_ops.iter_mut() {
            for shape in op_info.input_shapes.iter_mut().chain(op_info.output_shapes.iter_mut()) {
                for dim in shape.iter_mut() {
                    if let Some(&real) = shape_map.get(dim) {
                        *dim = real;
                    }
                }
            }
        }
        // Convert relative tensor IDs to global tensor IDs.
        // During fuse(), OperationIr uses relative IDs (0, 1, 2...) but the
        // HandleContainer uses global IDs. context.tensors maps relative → global.
        let id_map: HashMap<String, String> = context.tensors.iter()
            .map(|(rel_id, global_ir)| (format!("{}", rel_id), format!("{}", global_ir.id)))
            .collect();
        for op_info in fused_ops.iter_mut() {
            for id_str in op_info.input_ids.iter_mut().chain(op_info.output_ids.iter_mut()) {
                if let Some(global_id) = id_map.get(id_str.as_str()) {
                    *id_str = global_id.clone();
                }
            }
        }

        let caller = crate::trace::maybe_capture_caller();

        // Collect all global tensor IDs known to this fusion context.
        let all_global_ids: Vec<TensorId> = context.tensors.values()
            .map(|ir| ir.id)
            .collect();

        // Snapshot all existing handles BEFORE execution — these are potential
        // inputs. We clone them because execution may consume (remove) handles.
        let pre_existing = snapshot_existing_handles(context.handles, &all_global_ids);

        // Collect the set of output tensor IDs from the IR (already mapped to
        // global IDs). These are the tensors that will be registered during
        // execution.
        let ir_output_ids: HashSet<String> = fused_ops.iter()
            .flat_map(|op| op.output_ids.iter().cloned())
            .collect();

        let start = Instant::now();

        match &mut self.inner {
            CubeOptimization::ElementWise(op) => op.execute(context),
            CubeOptimization::Matmul(op) => {
                op.execute(context, |index| {
                    let operation = execution.operation_within_optimization(index);
                    Box::new(ProfiledFallbackWrapper { operation })
                });
            }
            CubeOptimization::Reduce(op) => {
                op.execute(context, |index| {
                    let operation = execution.operation_within_optimization(index);
                    Box::new(ProfiledFallbackWrapper { operation })
                });
            }
            CubeOptimization::ReduceBroadcasted(op) => {
                op.execute(context, |index| {
                    let operation = execution.operation_within_optimization(index);
                    Box::new(ProfiledFallbackWrapper { operation })
                });
            }
        }

        let duration = start.elapsed();

        // Look up output handles after execution using the IR-derived output IDs.
        let output_handles: Vec<(TensorId, CubeFusionHandle<R>)> = all_global_ids.iter()
            .filter(|id| ir_output_ids.contains(&format!("{}", id)))
            .filter_map(|id| context.handles.get_handle_ref(id).map(|h| (*id, h.clone())))
            .collect();
        let output_ids: HashSet<TensorId> = output_handles.iter().map(|(id, _)| *id).collect();

        // Inputs = handles that existed before execution but are not outputs.
        let input_handles: Vec<(TensorId, CubeFusionHandle<R>)> = pre_existing
            .into_iter()
            .filter(|(id, _)| !output_ids.contains(id))
            .collect();

        // Detect fallback ops: output not present in handle container after
        // execute() AND not consumed as input by any other fused op.
        let output_id_strs: HashSet<String> = output_handles.iter()
            .map(|(id, _)| format!("{}", id))
            .collect();
        let all_fused_input_ids: HashSet<String> = fused_ops.iter()
            .flat_map(|op| op.input_ids.iter().cloned())
            .collect();
        for op_info in fused_ops.iter_mut() {
            let has_handle = op_info.output_ids.iter()
                .any(|id| output_id_strs.contains(id));
            let is_consumed = op_info.output_ids.iter()
                .any(|id| all_fused_input_ids.contains(id));
            if !has_handle && !is_consumed {
                op_info.is_fallback = Some(true);
            }
        }

        // Capture data previews from input and output handles.
        let fusion_inputs = capture_fusion_handles(&input_handles);
        let fusion_outputs = capture_fusion_handles(&output_handles);

        record_fusion_execute(kind, num_ops, fused_ops, start, duration, caller, fusion_inputs, fusion_outputs, Vec::new());
    }

    fn to_state(&self) -> CubeOptimizationState {
        self.inner.to_opt_state()
    }

    fn from_state(device: &<ProfiledFusionRuntime<R> as FusionRuntime>::FusionDevice, state: CubeOptimizationState) -> Self {
        let inner = match state {
            CubeOptimizationState::ElementWise(s) => {
                CubeOptimization::ElementWise(ElemwiseOptimization::from_state(device, s))
            }
            CubeOptimizationState::Matmul(s) => {
                CubeOptimization::Matmul(MatmulOptimization::from_state(device, s))
            }
            CubeOptimizationState::Reduce(s) => {
                CubeOptimization::Reduce(ReduceOptimization::from_state(device, s))
            }
            CubeOptimizationState::ReduceBroadcasted(s) => {
                CubeOptimization::ReduceBroadcasted(ReduceBroadcastedOptimization::from_state(device, s))
            }
        };
        Self {
            inner,
            fused_op_info: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ProfiledFallbackWrapper
// ---------------------------------------------------------------------------

/// Wraps an `Operation<ProfiledFusionRuntime<R>>` so it can serve as a
/// `FallbackOperation<R>` for matmul/reduce optimizations.
struct ProfiledFallbackWrapper<R: CubeRuntime> {
    operation: Arc<dyn Operation<ProfiledFusionRuntime<R>>>,
}

impl<R: CubeRuntime> FallbackOperation<R> for ProfiledFallbackWrapper<R> {
    fn run(&self, context: &mut Context<'_, CubeFusionHandle<R>>) {
        self.operation.as_ref().execute(context.handles);
    }
}

// ---------------------------------------------------------------------------
// ProfiledFuser
// ---------------------------------------------------------------------------

/// Wraps an `OperationFuser<CubeOptimization<R>>` to produce
/// `ProfiledOptimization<R>` values, collecting per-op info during fusing.
struct ProfiledFuser<R: CubeRuntime> {
    inner: Box<dyn OperationFuser<CubeOptimization<R>>>,
    /// Accumulated op info for the current fusion group.
    fused_op_info: Vec<FusedOpInfo>,
}

impl<R: CubeRuntime> OperationFuser<ProfiledOptimization<R>> for ProfiledFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        // Capture op info before delegating.
        self.fused_op_info.push(op_ir_info(operation));
        self.inner.fuse(operation);
    }

    fn finish(&mut self) -> ProfiledOptimization<R> {
        let fused_op_info = std::mem::take(&mut self.fused_op_info);
        ProfiledOptimization {
            inner: self.inner.finish(),
            fused_op_info,
        }
    }

    fn reset(&mut self) {
        self.fused_op_info.clear();
        self.inner.reset();
    }

    fn status(&self) -> burn_fusion::FuserStatus {
        self.inner.status()
    }

    fn properties(&self) -> burn_fusion::FuserProperties {
        self.inner.properties()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<ProfiledOptimization<R>>> {
        Box::new(ProfiledFuser {
            inner: self.inner.clone_dyn(),
            fused_op_info: self.fused_op_info.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Fusion handle snapshot + data capture
// ---------------------------------------------------------------------------

/// Snapshot all existing handles in the container before execution.
/// We clone them because execution may consume (remove) handles via `get_handle`.
#[cfg(feature = "trace-data")]
fn snapshot_existing_handles<R: CubeRuntime>(
    handles: &burn_ir::HandleContainer<CubeFusionHandle<R>>,
    global_ids: &[TensorId],
) -> Vec<(TensorId, CubeFusionHandle<R>)> {
    global_ids.iter()
        .filter_map(|id| handles.get_handle_ref(id).map(|h| (*id, h.clone())))
        .collect()
}

#[cfg(not(feature = "trace-data"))]
fn snapshot_existing_handles<R: CubeRuntime>(
    _handles: &burn_ir::HandleContainer<CubeFusionHandle<R>>,
    _global_ids: &[TensorId],
) -> Vec<(TensorId, CubeFusionHandle<R>)> {
    Vec::new()
}

/// Reconstruct tensor shape from contiguous row-major strides + total element count.
///
/// For a contiguous tensor with shape `[d0, d1, ..., dn]`, the strides are
/// `[d1*d2*...*dn, d2*...*dn, ..., dn, 1]`. We recover each dimension by
/// dividing consecutive strides.
#[cfg(feature = "trace-data")]
fn shape_from_strides(strides: &[usize], total_elems: usize) -> Vec<usize> {
    if strides.is_empty() {
        return vec![total_elems];
    }
    let ndim = strides.len();
    let mut shape = vec![0usize; ndim];
    if strides[0] > 0 {
        shape[0] = total_elems / strides[0];
    } else {
        shape[0] = total_elems;
    }
    for i in 1..ndim {
        if strides[i] > 0 {
            shape[i] = strides[i - 1] / strides[i];
        } else {
            shape[i] = 1;
        }
    }
    shape
}

/// Capture data previews from a list of `(TensorId, CubeFusionHandle)` pairs.
/// Used for both inputs (pre-existing handles) and outputs (registration log).
#[cfg(feature = "trace-data")]
fn capture_fusion_handles<R: CubeRuntime>(
    handles: &[(TensorId, CubeFusionHandle<R>)],
) -> Vec<FusionTensorData> {
    use burn_backend::TensorData;

    let limit = crate::trace::data_capture_limit();
    if limit == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();
    for (id, handle) in handles {
        let dtype = handle.dtype;
        let dtype_size = dtype.size();
        if dtype_size == 0 {
            continue;
        }

        let Ok(bytes) = handle.client.read_one(handle.handle.clone()) else {
            continue;
        };

        let total_elems = bytes.len() / dtype_size;
        let shape = shape_from_strides(&handle.strides, total_elems);

        let data = TensorData::from_bytes(
            bytes,
            burn_backend::Shape::new_raw(shape.iter().copied().collect()),
            dtype,
        );
        let preview: Vec<f64> = data.iter::<f64>().take(limit).collect();

        result.push(FusionTensorData {
            shape,
            dtype: format!("{:?}", dtype),
            data: preview,
            tensor_id: Some(format!("{}", id)),
        });
    }

    result
}

#[cfg(not(feature = "trace-data"))]
fn capture_fusion_handles<R: CubeRuntime>(
    _handles: &[(TensorId, CubeFusionHandle<R>)],
) -> Vec<FusionTensorData> {
    Vec::new()
}
