use burn_backend::TensorMetadata;
use serde::Serialize;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(feature = "trace-data")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Category of a traced operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OpCategory {
    Float,
    Int,
    Bool,
    Module,
    Activation,
    Quantized,
    Transaction,
    Sync,
    Fusion,
    Marker,
}

impl OpCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            OpCategory::Float => "float",
            OpCategory::Int => "int",
            OpCategory::Bool => "bool",
            OpCategory::Module => "module",
            OpCategory::Activation => "activation",
            OpCategory::Quantized => "quantized",
            OpCategory::Transaction => "transaction",
            OpCategory::Sync => "sync",
            OpCategory::Fusion => "fusion",
            OpCategory::Marker => "marker",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            OpCategory::Float => "#4285f4",
            OpCategory::Int => "#34a853",
            OpCategory::Bool => "#fbbc04",
            OpCategory::Module => "#ea4335",
            OpCategory::Activation => "#ff6d01",
            OpCategory::Quantized => "#46bdc6",
            OpCategory::Transaction => "#7baaf7",
            OpCategory::Sync => "#ff0000",
            OpCategory::Fusion => "#9c27b0",
            OpCategory::Marker => "#00e5ff",
        }
    }
}

/// The kind of fused optimization that was executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FusionKind {
    ElementWise,
    Matmul,
    Reduce,
    ReduceBroadcasted,
}

/// Information about a single operation within a fusion block.
#[derive(Debug, Clone, Serialize)]
pub struct FusedOpInfo {
    /// Human-readable name like "float::Add", "float::Exp", "module::Conv2d".
    pub name: String,
    /// Shapes of input tensors.
    pub input_shapes: Vec<Vec<usize>>,
    /// Shapes of output tensors.
    pub output_shapes: Vec<Vec<usize>>,
    /// Relative tensor IDs of inputs (parallel to `input_shapes`).
    /// These are stable across executions and match the context's HandleContainer keys.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_ids: Vec<String>,
    /// Relative tensor IDs of outputs (parallel to `output_shapes`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub output_ids: Vec<String>,
    /// True if this op was NOT executed by the fused kernel but instead ran
    /// as a separate kernel launch (fallback) by the framework.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_fallback: Option<bool>,
}

/// Data captured from one input or output tensor of a fusion block.
#[derive(Debug, Clone, Serialize)]
pub struct FusionTensorData {
    /// Shape of this tensor (reconstructed from strides).
    pub shape: Vec<usize>,
    /// Element type name (e.g. "F32", "Bool(U32)").
    pub dtype: String,
    /// Preview of the first N values as f64.
    pub data: Vec<f64>,
    /// Tensor ID linking this data to a specific fused op input/output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_id: Option<String>,
}

/// A single traced event.
#[derive(Debug, Clone, Serialize)]
pub struct TraceEvent {
    pub name: String,
    pub category: OpCategory,
    /// Microseconds since trace start.
    pub start_us: f64,
    /// Duration in microseconds.
    pub duration_us: f64,
    /// Sequential operation index.
    pub op_index: usize,
    /// Estimated output tensor memory in bytes (for allocation-like ops).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    /// Input tensor shapes (when available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_shapes: Option<Vec<Vec<usize>>>,
    /// Output tensor shape (when available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_shape: Option<Vec<usize>>,
    /// The kind of fusion optimization (only set for fusion events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_kind: Option<FusionKind>,
    /// Number of operations fused in this optimization (only set for fusion events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_fused_ops: Option<usize>,
    /// Details of each fused operation (only set for fusion events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused_ops: Option<Vec<FusedOpInfo>>,
    /// Caller location from backtrace (only with `trace-caller` feature).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller: Option<String>,
    /// Preview of the first N output tensor values (only with `trace-data` feature).
    /// Deprecated: use `fusion_outputs` for fusion events. Still used for non-fusion ops.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_preview: Option<Vec<f64>>,
    /// Shape of the tensor whose data is previewed (may differ from output_shape for fusions).
    /// Deprecated: use `fusion_outputs` for fusion events. Still used for non-fusion ops.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_shape: Option<Vec<usize>>,
    /// All input tensors of a fusion block with their shapes and data previews.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_inputs: Option<Vec<FusionTensorData>>,
    /// All output tensors of a fusion block with their shapes and data previews.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_outputs: Option<Vec<FusionTensorData>>,
    /// Whether this is a span (has duration) or an instant marker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_span: Option<bool>,
    /// User-provided debug string attached to a marker/span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_data: Option<String>,
    /// User-provided binary data attached to a marker/span (base64-encoded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_data: Option<String>,
    /// Whether this operation implicitly synchronizes the GPU (e.g. `into_data`
    /// flushes the fusion queue and blocks on GPU readback).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_sync: Option<bool>,
    /// True if this standalone event is a fusion fallback — an operation that
    /// was part of a fusion pattern but executed as a separate kernel launch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_fusion_fallback: Option<bool>,
    /// Compiled kernel source code(s) for this fusion event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_sources: Option<Vec<String>>,
}

/// Collects trace events from profiled operations.
pub struct TraceCollector {
    events: Vec<TraceEvent>,
    epoch: Option<Instant>,
    op_counter: usize,
}

impl TraceCollector {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            epoch: None,
            op_counter: 0,
        }
    }

    fn ensure_epoch(&mut self) -> Instant {
        *self.epoch.get_or_insert_with(Instant::now)
    }

    pub fn record_full(&mut self, name: &str, category: OpCategory, start: Instant, duration: Duration,
        memory_bytes: Option<u64>, input_shapes: Option<Vec<Vec<usize>>>, output_shape: Option<Vec<usize>>,
        caller: Option<String>, data_preview: Option<Vec<f64>>,
    ) {
        let epoch = self.ensure_epoch();
        let start_us = start.duration_since(epoch).as_nanos() as f64 / 1000.0;
        let duration_us = duration.as_nanos() as f64 / 1000.0;
        let op_index = self.op_counter;
        self.op_counter += 1;
        // For regular ops, data_shape equals output_shape (only set when data_preview exists).
        let data_shape = if data_preview.is_some() { output_shape.clone() } else { None };
        self.events.push(TraceEvent {
            name: name.to_string(),
            category,
            start_us,
            duration_us,
            op_index,
            memory_bytes,
            input_shapes,
            output_shape,
            fusion_kind: None,
            num_fused_ops: None,
            fused_ops: None,
            caller,
            data_preview,
            data_shape,
            fusion_inputs: None,
            fusion_outputs: None,
            is_span: None,
            debug_data: None,
            binary_data: None,
            is_sync: None,
            is_fusion_fallback: None,
            kernel_sources: None,
        });
    }

    pub fn record(&mut self, name: &str, category: OpCategory, start: Instant, duration: Duration, memory_bytes: Option<u64>, input_shapes: Option<Vec<Vec<usize>>>, output_shape: Option<Vec<usize>>) {
        self.record_full(name, category, start, duration, memory_bytes, input_shapes, output_shape, None, None);
    }

    /// Record an operation that implicitly synchronizes the GPU (e.g. `into_data`).
    pub fn record_sync_op(&mut self, name: &str, category: OpCategory, start: Instant, duration: Duration,
        memory_bytes: Option<u64>, output_shape: Option<Vec<usize>>, data_preview: Option<Vec<f64>>,
    ) {
        self.record_full(name, category, start, duration, memory_bytes, None, output_shape, None, data_preview);
        // Mark the last event as an implicit sync
        if let Some(last) = self.events.last_mut() {
            last.is_sync = Some(true);
        }
    }

    pub fn record_fusion(&mut self, kind: FusionKind, num_fused_ops: usize, fused_ops: Vec<FusedOpInfo>, start: Instant, duration: Duration, caller: Option<String>, fusion_inputs: Vec<FusionTensorData>, fusion_outputs: Vec<FusionTensorData>, kernel_sources: Vec<String>) {
        let epoch = self.ensure_epoch();
        let start_us = start.duration_since(epoch).as_nanos() as f64 / 1000.0;
        let duration_us = duration.as_nanos() as f64 / 1000.0;
        let name = match kind {
            FusionKind::ElementWise => "fusion::elementwise",
            FusionKind::Matmul => "fusion::matmul",
            FusionKind::Reduce => "fusion::reduce",
            FusionKind::ReduceBroadcasted => "fusion::reduce_broadcasted",
        };
        let op_index = self.op_counter;
        self.op_counter += 1;
        let fused_ops = if fused_ops.is_empty() { None } else { Some(fused_ops) };
        let fusion_inputs = if fusion_inputs.is_empty() { None } else { Some(fusion_inputs) };
        let fusion_outputs = if fusion_outputs.is_empty() { None } else { Some(fusion_outputs) };
        let kernel_sources = if kernel_sources.is_empty() { None } else { Some(kernel_sources) };
        self.events.push(TraceEvent {
            name: name.to_string(),
            category: OpCategory::Fusion,
            start_us,
            duration_us,
            op_index,
            memory_bytes: None,
            input_shapes: None,
            output_shape: None,
            fusion_kind: Some(kind),
            num_fused_ops: Some(num_fused_ops),
            fused_ops,
            caller,
            data_preview: None,
            data_shape: None,
            fusion_inputs,
            fusion_outputs,
            is_span: None,
            debug_data: None,
            binary_data: None,
            is_sync: None,
            is_fusion_fallback: None,
            kernel_sources,
        });
    }

    /// Record a user marker (instant event or span).
    pub fn record_marker(
        &mut self,
        name: &str,
        start: Instant,
        duration: Duration,
        is_span: bool,
        debug_data: Option<String>,
        binary_data: Option<Vec<u8>>,
        tensors: Vec<FusionTensorData>,
        caller: Option<String>,
    ) {
        let epoch = self.ensure_epoch();
        let start_us = start.duration_since(epoch).as_nanos() as f64 / 1000.0;
        let duration_us = duration.as_nanos() as f64 / 1000.0;
        let op_index = self.op_counter;
        self.op_counter += 1;
        let binary_b64 = binary_data.map(|b| {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(&b)
        });
        let fusion_outputs = if tensors.is_empty() { None } else { Some(tensors) };
        self.events.push(TraceEvent {
            name: name.to_string(),
            category: OpCategory::Marker,
            start_us,
            duration_us,
            op_index,
            memory_bytes: None,
            input_shapes: None,
            output_shape: None,
            fusion_kind: None,
            num_fused_ops: None,
            fused_ops: None,
            caller,
            data_preview: None,
            data_shape: None,
            fusion_inputs: None,
            fusion_outputs,
            is_span: Some(is_span),
            debug_data,
            binary_data: binary_b64,
            is_sync: None,
            is_fusion_fallback: None,
            kernel_sources: None,
        });
    }

    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.epoch = None;
        self.op_counter = 0;
    }
}

static GLOBAL_TRACER: Mutex<Option<TraceCollector>> = Mutex::new(None);


fn with_tracer<F, R>(f: F) -> R
where
    F: FnOnce(&mut TraceCollector) -> R,
{
    let mut guard = GLOBAL_TRACER.lock().unwrap();
    let tracer = guard.get_or_insert_with(TraceCollector::new);
    f(tracer)
}

/// Start tracing. Clears any previous trace data.
///
/// Also configures CubeCL's compilation logger (via `CUBECL_DEBUG_LOG` env var)
/// to write full kernel sources to a temp file so they can be extracted later.
/// This must be called **before** any CubeCL device is initialised.
pub fn start_tracing() {
    // Set up CubeCL compilation logging to a temp file so we can extract
    // compiled kernel sources in finish_tracing(). The env var must be set
    // before GlobalConfig is first read (i.e. before device init).
    #[cfg(feature = "trace-data")]
    {
        let log_path = std::env::temp_dir().join("burn_tracing_backend_cubecl.log");
        // Truncate the file so we only see compilations from this run.
        let _ = std::fs::write(&log_path, "");
        // SAFETY: This is called at program start before any threads are spawned
        // that might read this env var. CubeCL reads it during GlobalConfig::get()
        // which happens at first device init — after start_tracing().
        unsafe {
            std::env::set_var("CUBECL_DEBUG_LOG", log_path.to_str().unwrap_or("/tmp/burn_tracing_backend_cubecl.log"));
        }
    }

    with_tracer(|t| t.clear());
}

/// Stop tracing and return all collected events.
///
/// Post-processes to remove duplicate events caused by the fusion backend
/// re-executing fused ops through the inner profiler for handle materialization.
/// Also parses the CubeCL compilation log to extract kernel sources and attach
/// them to fusion events.
pub fn finish_tracing() -> Vec<TraceEvent> {
    #[allow(unused_mut)]
    let mut events = with_tracer(|t| {
        let events = deduplicate_fused_ops(t.events());
        t.clear();
        events
    });

    // Parse kernel sources from the CubeCL compilation log and attach them
    // to fusion events. Each fusion event gets all compiled kernel sources
    // (there's typically one kernel per fusion block, sometimes two for matmul).
    #[cfg(feature = "trace-data")]
    {
        let kernel_sources = parse_cubecl_compilation_log();
        if !kernel_sources.is_empty() {
            attach_kernel_sources(&mut events, &kernel_sources);
        }
    }

    events
}

/// Remove standalone op events that duplicate ops already recorded inside
/// a fusion block (caused by the Fusion backend re-executing fused ops
/// through the inner Profiler for output materialization / fallback).
///
/// Duplicates can appear both before AND after the fusion event (fallback
/// ops run during execute, materialization ops run after). We identify
/// duplicates by matching op names within each sync-delimited segment.
///
/// When a standalone duplicate has data (data_preview/data_shape) that the
/// fusion event doesn't already have, the data is merged into the fusion
/// event's `fusion_outputs` before the standalone event is removed.
fn deduplicate_fused_ops(events: &[TraceEvent]) -> Vec<TraceEvent> {
    let mut result = Vec::with_capacity(events.len());
    let mut segment_start = 0;

    loop {
        let segment_end = events[segment_start..]
            .iter()
            .position(|e| e.category == OpCategory::Sync)
            .map(|p| segment_start + p + 1)
            .unwrap_or(events.len());

        let segment = &events[segment_start..segment_end];

        // Build two sets of counts:
        // - merge_counts/per_merge_counts: ALL ops (including fallbacks) — for data merging
        // - skip_counts: non-fallback ops only — for removing duplicate standalone events
        let mut merge_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut per_merge_counts: std::collections::HashMap<usize, std::collections::HashMap<String, usize>> =
            std::collections::HashMap::new();
        let mut skip_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut fallback_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Per-fusion FIFO of output tensor IDs for data merge.
        let mut per_fusion_output_ids: std::collections::HashMap<usize, std::collections::HashMap<String, std::collections::VecDeque<Option<String>>>> =
            std::collections::HashMap::new();
        for (i, e) in segment.iter().enumerate() {
            if let Some(ref fused_ops) = e.fused_ops {
                let mc = per_merge_counts.entry(i).or_default();
                let oid_map = per_fusion_output_ids.entry(i).or_default();
                for op in fused_ops {
                    let key = op.name.replace("::", "_").to_lowercase();
                    // Always count for merging
                    *merge_counts.entry(key.clone()).or_insert(0) += 1;
                    *mc.entry(key.clone()).or_insert(0) += 1;
                    let first_output_id = op.output_ids.first().cloned();
                    oid_map.entry(key.clone()).or_default().push_back(first_output_id);
                    if op.is_fallback == Some(true) {
                        fallback_names.insert(key);
                    } else {
                        // Only count non-fallbacks for skip (removal)
                        *skip_counts.entry(op.name.replace("::", "_").to_lowercase()).or_insert(0) += 1;
                    }
                }
            }
        }

        // First pass: identify standalone events and merge their data into
        // the parent fusion event's fusion_outputs (with tensor_id).
        let mut merge_data: std::collections::HashMap<usize, Vec<FusionTensorData>> =
            std::collections::HashMap::new();
        {
            let mut counts_tmp = merge_counts.clone();
            let mut per_counts_tmp = per_merge_counts.clone();
            for e in segment {
                if e.category == OpCategory::Fusion || e.category == OpCategory::Sync {
                    continue;
                }
                let key = e.name.to_lowercase();
                if let Some(count) = counts_tmp.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        // Find which fusion event should receive this data.
                        if let (Some(preview), Some(shape)) = (&e.data_preview, &e.data_shape) {
                            if !preview.is_empty() {
                                for (&fi, fc) in per_counts_tmp.iter_mut() {
                                    if let Some(c) = fc.get_mut(&key) {
                                        if *c > 0 {
                                            *c -= 1;
                                            // Look up tensor_id from fused op's output_ids
                                            let tensor_id = per_fusion_output_ids
                                                .get_mut(&fi)
                                                .and_then(|m| m.get_mut(&key))
                                                .and_then(|ids| ids.pop_front())
                                                .flatten();
                                            merge_data.entry(fi).or_default().push(FusionTensorData {
                                                shape: shape.clone(),
                                                dtype: String::new(),
                                                data: preview.clone(),
                                                tensor_id,
                                            });
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Second pass: emit events, merging data into fusion events and
        // skipping duplicates.
        for (i, e) in segment.iter().enumerate() {
            if e.category == OpCategory::Fusion || e.category == OpCategory::Sync {
                let mut event = e.clone();
                if let Some(extra_outputs) = merge_data.remove(&i) {
                    let outputs = event.fusion_outputs.get_or_insert_with(Vec::new);
                    outputs.extend(extra_outputs);
                }
                result.push(event);
            } else {
                let key = e.name.to_lowercase();
                if let Some(count) = skip_counts.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        continue;
                    }
                }
                let mut event = e.clone();
                if fallback_names.contains(&key) {
                    event.is_fusion_fallback = Some(true);
                }
                result.push(event);
            }
        }

        if segment_end >= events.len() {
            break;
        }
        segment_start = segment_end;
    }

    result
}

// ---------------------------------------------------------------------------
// CubeCL compilation log parsing (only when trace-data feature is enabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "trace-data")]
/// A parsed kernel entry from the CubeCL compilation log.
struct ParsedKernel {
    /// Full type name from the `name:` line (first line only, trimmed).
    name: String,
    /// Shader source code.
    source: String,
    /// For ElemwiseFuse kernels: the non-Assign op names extracted from the
    /// `ops:` list in the KernelId info section.
    ops_signature: Vec<String>,
}

#[cfg(feature = "trace-data")]
/// Parse the CubeCL compilation log file to extract kernel names and sources.
///
/// The Full-level log format uses:
/// ```text
/// [START_KERNEL_COMPILATION]
/// name: <kernel_name>
/// ...
/// source:
/// ```<lang>
/// <source code>
/// ```
/// [END_KERNEL_COMPILATION]
/// ```
fn parse_cubecl_compilation_log() -> Vec<ParsedKernel> {
    let log_path = std::env::temp_dir().join("burn_tracing_backend_cubecl.log");
    let content = match std::fs::read_to_string(&log_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut kernels = Vec::new();
    let mut lines = content.lines().peekable();

    while let Some(line) = lines.next() {
        if line.trim() == "[START_KERNEL_COMPILATION]" {
            let mut name = String::new();
            let mut source = String::new();
            let mut id_text = String::new();
            let mut in_source = false;
            let mut in_id = false;

            for inner_line in lines.by_ref() {
                let trimmed = inner_line.trim();
                if trimmed == "[END_KERNEL_COMPILATION]" {
                    break;
                }
                if in_source {
                    if trimmed.starts_with("```") && !source.is_empty() {
                        in_source = false;
                        continue;
                    }
                    if !source.is_empty() {
                        source.push('\n');
                    }
                    source.push_str(inner_line);
                } else if in_id {
                    if trimmed == "}" && inner_line.starts_with('}') {
                        // Top-level closing brace of KernelId
                        id_text.push('\n');
                        id_text.push_str(inner_line);
                        in_id = false;
                    } else {
                        id_text.push('\n');
                        id_text.push_str(inner_line);
                    }
                } else if let Some(n) = trimmed.strip_prefix("name: ") {
                    name = n.to_string();
                } else if trimmed.starts_with("id: KernelId") {
                    in_id = true;
                    id_text = trimmed.to_string();
                } else if trimmed.starts_with("```") {
                    in_source = true;
                }
            }

            if !source.is_empty() {
                let ops_signature = extract_ops_signature(&id_text);
                kernels.push(ParsedKernel { name, source, ops_signature });
            }
        }
    }

    kernels
}

#[cfg(feature = "trace-data")]
/// Extract op names from the `ops:` list in a KernelId's info section.
///
/// The ops list in the kernel ID looks like:
/// ```text
///             ops: [
///                 Assign(
///                     UnaryFuseArgs { ... },
///                 ),
///                 Add(
///                     BinaryFuseArgs { ... },
///                 ),
///             ],
/// ```
/// We find the `ops: [` line, note its indentation, and then look for
/// op entries at exactly one more indentation level.
fn extract_ops_signature(id_text: &str) -> Vec<String> {
    let mut ops = Vec::new();
    let mut in_ops = false;
    // Track all bracket types to know nesting depth within the ops list.
    let mut depth: i32 = 0;

    for line in id_text.lines() {
        let trimmed = line.trim();

        if !in_ops {
            if trimmed.starts_with("ops: [") {
                in_ops = true;
                depth = 0;
                // Count opening brackets on this line.
                for ch in trimmed.chars() {
                    match ch {
                        '[' | '(' | '{' => depth += 1,
                        ']' | ')' | '}' => depth -= 1,
                        _ => {}
                    }
                }
                continue;
            }
        } else {
            // Compute depth before processing this line.
            let pre_depth = depth;
            for ch in trimmed.chars() {
                match ch {
                    '[' | '(' | '{' => depth += 1,
                    ']' | ')' | '}' => depth -= 1,
                    _ => {}
                }
            }
            // Depth 0 means we've closed the ops array.
            if depth <= 0 {
                return ops;
            }
            // Op entry lines are at pre_depth == 1 and start with a capitalized
            // word followed by ( or {. The indentation should be ops_indent + 4
            // (one indent level deeper), but we rely on depth instead.
            if pre_depth == 1 {
                if let Some(op_name) = trimmed.split(&['(', ' ', '{'][..]).next() {
                    if !op_name.is_empty()
                        && op_name != "Assign"
                        && op_name != "],"
                        && op_name.chars().next().map_or(false, |c| c.is_uppercase())
                    {
                        ops.push(op_name.to_string());
                    }
                }
            }
        }
    }
    ops
}

#[cfg(feature = "trace-data")]
/// Build a matching signature from a fusion event's fused ops.
///
/// Strips the category prefix (e.g. `float::`) and normalizes op names to
/// match the kernel ID op names (e.g. `LowerElem` → `Lower`,
/// `MulScalar` → `Mul`, `MaskFill` → `ConditionalAssign`).
/// Skips structural ops like `Reshape`, `Cast`, `Matmul`, `Mean` that
/// don't appear as elemwise fusion ops.
fn fusion_ops_signature(fused_ops: &[FusedOpInfo]) -> Vec<String> {
    let mut sig = Vec::new();
    for op in fused_ops {
        // Strip category prefix: "float::Add" → "Add"
        let base = op.name.split("::").last().unwrap_or(&op.name);
        // Normalize names to match kernel ID convention
        let normalized = match base {
            // Scalar variants map to the same kernel op
            "AddScalar" => "Add",
            "SubScalar" => "Sub",
            "MulScalar" => "Mul",
            "DivScalar" => "Div",
            "PowfScalar" => "Powf",
            "MaxScalar" | "MaxPair" => "Max",
            "MinScalar" | "MinPair" => "Min",
            // Comparison elem variants
            "LowerElem" => "Lower",
            "GreaterElem" => "Greater",
            "LowerEqualElem" => "LowerEqual",
            "GreaterEqualElem" => "GreaterEqual",
            "EqualElem" => "Equal",
            "NotEqualElem" => "NotEqual",
            // MaskFill becomes ConditionalAssign in the kernel
            "MaskFill" | "MaskWhere" => "ConditionalAssign",
            // These are structural ops handled by Assign in the kernel — skip
            "Reshape" | "Cast" | "SwapDims" | "Permute" | "Repeat" | "ExpandAs" => continue,
            // These are not elemwise fusion ops — skip
            "Matmul" | "Mean" | "Sum" | "Prod" | "ArgMax" | "ArgMin" => continue,
            other => other,
        };
        sig.push(normalized.to_string());
    }
    sig
}

#[cfg(feature = "trace-data")]
/// Attach parsed kernel sources to fusion events using multi-pass matching:
///
/// 1. **ElemwiseFuse** — matched by ops signature (ops list in kernel ID vs fused ops)
/// 2. **MatmulEntry** — matched to `FusionKind::Matmul` events by sequential order
/// 3. **ReduceKernelBroadcasted** — matched to `FusionKind::ReduceBroadcasted` events by order
/// 4. **ReduceKernelFused** — matched to `FusionKind::Reduce` events by order
fn attach_kernel_sources(events: &mut [TraceEvent], kernels: &[ParsedKernel]) {
    // Collect fusion events with their signatures.
    let fusion_indices: Vec<usize> = events.iter()
        .enumerate()
        .filter(|(_, e)| e.category == OpCategory::Fusion)
        .map(|(i, _)| i)
        .collect();

    if fusion_indices.is_empty() {
        return;
    }

    // Separate kernels into ElemwiseFuse (with ops signature) and others.
    // ElemwiseFuse kernels can be matched by their ops signature.
    // Other fusion kernels (MatmulEntry, ReduceKernel, etc.) are matched
    // to non-elemwise fusion events by sequential order.

    // Track which kernels have been matched.
    let mut kernel_matched = vec![false; kernels.len()];

    // First pass: match ElemwiseFuse kernels to elementwise fusion events
    // by ops signature.
    for &event_idx in &fusion_indices {
        let e = &events[event_idx];
        let fusion_sig = e.fused_ops.as_ref()
            .map(|ops| fusion_ops_signature(ops))
            .unwrap_or_default();
        if fusion_sig.is_empty() {
            continue;
        }

        // Find matching ElemwiseFuse kernel by ops signature.
        for (ki, kernel) in kernels.iter().enumerate() {
            if kernel_matched[ki] {
                continue;
            }
            if !kernel.name.contains("ElemwiseFuse") {
                continue;
            }
            if kernel.ops_signature == fusion_sig {
                let sources = events[event_idx].kernel_sources.get_or_insert_with(Vec::new);
                sources.push(kernel.source.clone());
                kernel_matched[ki] = true;
                break;
            }
        }
    }

    // Second pass: match MatmulEntry kernels to matmul fusion events by order.
    let matmul_fusions: Vec<usize> = fusion_indices.iter()
        .filter(|&&idx| matches!(events[idx].fusion_kind, Some(FusionKind::Matmul)))
        .copied()
        .collect();
    let matmul_kernels: Vec<usize> = kernels.iter().enumerate()
        .filter(|(i, k)| !kernel_matched[*i] && k.name.contains("MatmulEntry"))
        .map(|(i, _)| i)
        .collect();
    let mut matmul_ki = 0;
    for &event_idx in &matmul_fusions {
        if matmul_ki < matmul_kernels.len() {
            let ki = matmul_kernels[matmul_ki];
            let sources = events[event_idx].kernel_sources.get_or_insert_with(Vec::new);
            sources.push(kernels[ki].source.clone());
            kernel_matched[ki] = true;
            matmul_ki += 1;
        }
    }

    // Third pass: match ReduceKernelBroadcasted kernels to ReduceBroadcasted fusion events.
    let reduce_bc_fusions: Vec<usize> = fusion_indices.iter()
        .filter(|&&idx| matches!(events[idx].fusion_kind, Some(FusionKind::ReduceBroadcasted)))
        .copied()
        .collect();
    let reduce_bc_kernels: Vec<usize> = kernels.iter().enumerate()
        .filter(|(i, k)| !kernel_matched[*i] && k.name.contains("ReduceKernelBroadcasted"))
        .map(|(i, _)| i)
        .collect();
    let mut reduce_bc_ki = 0;
    for &event_idx in &reduce_bc_fusions {
        if reduce_bc_ki < reduce_bc_kernels.len() {
            let ki = reduce_bc_kernels[reduce_bc_ki];
            let sources = events[event_idx].kernel_sources.get_or_insert_with(Vec::new);
            sources.push(kernels[ki].source.clone());
            kernel_matched[ki] = true;
            reduce_bc_ki += 1;
        }
    }

    // Fourth pass: match ReduceKernelFused kernels to Reduce fusion events.
    let reduce_fusions: Vec<usize> = fusion_indices.iter()
        .filter(|&&idx| matches!(events[idx].fusion_kind, Some(FusionKind::Reduce)))
        .copied()
        .collect();
    let reduce_kernels: Vec<usize> = kernels.iter().enumerate()
        .filter(|(i, k)| !kernel_matched[*i] && (
            k.name.contains("ReduceKernelFused") ||
            k.name.contains("SharedSumKernel") ||
            k.name.contains("ReduceFuse")
        ))
        .map(|(i, _)| i)
        .collect();
    let mut reduce_ki = 0;
    for &event_idx in &reduce_fusions {
        if reduce_ki < reduce_kernels.len() {
            let ki = reduce_kernels[reduce_ki];
            let sources = events[event_idx].kernel_sources.get_or_insert_with(Vec::new);
            sources.push(kernels[ki].source.clone());
            kernel_matched[ki] = true;
            reduce_ki += 1;
        }
    }
}

/// Get a snapshot of current events without clearing.
pub fn snapshot_events() -> Vec<TraceEvent> {
    with_tracer(|t| t.events().to_vec())
}

/// Record a completed operation.
pub(crate) fn record_op(name: &str, category: OpCategory, start: Instant, duration: Duration) {
    with_tracer(|t| t.record(name, category, start, duration, None, None, None));
}

/// Record a completed operation with memory info.
#[allow(dead_code)]
pub(crate) fn record_op_with_memory(name: &str, category: OpCategory, start: Instant, duration: Duration, memory_bytes: u64) {
    with_tracer(|t| t.record(name, category, start, duration, Some(memory_bytes), None, None));
}

/// Record an operation that implicitly synchronizes the GPU (flushes fusion queue + blocks on readback).
/// These ops (e.g. `into_data`) do not allocate GPU memory — they read GPU→CPU.
pub(crate) fn record_op_sync(name: &str, category: OpCategory, start: Instant, duration: Duration,
    output_shape: Option<Vec<usize>>, data_preview: Option<Vec<f64>>,
) {
    with_tracer(|t| t.record_sync_op(name, category, start, duration, None, output_shape, data_preview));
}

/// Extract a data preview and shape from `TensorData` (for `into_data` ops).
/// Returns `(shape, data_preview)` when the `trace-data` feature is enabled.
#[cfg(feature = "trace-data")]
pub(crate) fn extract_tensor_data_preview(data: &burn_backend::TensorData) -> (Vec<usize>, Option<Vec<f64>>) {
    let shape: Vec<usize> = data.shape.iter().copied().collect();
    let limit = data_capture_limit();
    if limit == 0 {
        return (shape, None);
    }
    let preview = extract_preview(data, limit);
    (shape, Some(preview))
}

#[cfg(not(feature = "trace-data"))]
pub(crate) fn extract_tensor_data_preview(data: &burn_backend::TensorData) -> (Vec<usize>, Option<Vec<f64>>) {
    let shape: Vec<usize> = data.shape.iter().copied().collect();
    (shape, None)
}

/// Record a creation-like op (random, zeros, ones, ...) for a **float** tensor,
/// including caller and data capture when the corresponding features are enabled.
pub(crate) fn record_float_creation<B: burn_backend::Backend>(
    name: &str, category: OpCategory, start: Instant, duration: Duration,
    memory_bytes: u64, shape: Vec<usize>, tensor: &burn_backend::tensor::FloatTensor<B>,
) {
    let caller = maybe_capture_caller();
    let data_preview = maybe_capture_float_data::<B>(tensor);
    with_tracer(|t| t.record_full(name, category, start, duration, Some(memory_bytes), None, Some(shape), caller, data_preview));
}

/// Record a creation-like op for an **int** tensor.
pub(crate) fn record_int_creation<B: burn_backend::Backend>(
    name: &str, category: OpCategory, start: Instant, duration: Duration,
    memory_bytes: u64, shape: Vec<usize>, tensor: &burn_backend::tensor::IntTensor<B>,
) {
    let caller = maybe_capture_caller();
    let data_preview = maybe_capture_int_data::<B>(tensor);
    with_tracer(|t| t.record_full(name, category, start, duration, Some(memory_bytes), None, Some(shape), caller, data_preview));
}

/// Record a creation-like op for a **bool** tensor.
pub(crate) fn record_bool_creation<B: burn_backend::Backend>(
    name: &str, category: OpCategory, start: Instant, duration: Duration,
    memory_bytes: u64, shape: Vec<usize>, tensor: &burn_backend::tensor::BoolTensor<B>,
) {
    let caller = maybe_capture_caller();
    let data_preview = maybe_capture_bool_data::<B>(tensor);
    with_tracer(|t| t.record_full(name, category, start, duration, Some(memory_bytes), None, Some(shape), caller, data_preview));
}

/// Compute the estimated memory in bytes for a tensor.
pub(crate) fn tensor_bytes(tensor: &impl TensorMetadata) -> u64 {
    tensor.shape().num_elements() as u64 * tensor.dtype().size() as u64
}

/// Extract shape as Vec<usize> from a tensor.
pub(crate) fn tensor_shape(tensor: &impl TensorMetadata) -> Vec<usize> {
    tensor.shape().iter().copied().collect()
}

/// Autoref-based specialization to extract shapes from tensor args.
///
/// Usage in macros: `ShapeProbe(&arg).maybe_shape()`
///
/// For `T: TensorMetadata`: `ShapeProbe<&T>` has an inherent `maybe_shape()` → returns Some(shape).
/// For other types: no inherent method → Rust finds `MaybeShapeFallback` trait method → returns None.
pub(crate) struct ShapeProbe<T>(pub T);

// Inherent impl: wins over trait methods in method resolution
impl<T: TensorMetadata> ShapeProbe<&T> {
    pub fn maybe_shape(&self) -> Option<Vec<usize>> {
        Some(self.0.shape().iter().copied().collect())
    }
}

// Trait fallback: only reached when no inherent method matches
pub(crate) trait MaybeShapeFallback {
    fn maybe_shape(&self) -> Option<Vec<usize>>;
}
impl<T> MaybeShapeFallback for ShapeProbe<T> {
    fn maybe_shape(&self) -> Option<Vec<usize>> {
        None
    }
}

/// RAII guard that records timing when dropped.
pub(crate) struct TraceGuard {
    name: &'static str,
    category: OpCategory,
    start: Instant,
    caller: Option<String>,
}

impl TraceGuard {
    pub fn new(name: &'static str, category: OpCategory) -> Self {
        Self {
            name,
            category,
            start: Instant::now(),
            caller: maybe_capture_caller(),
        }
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        with_tracer(|t| t.record_full(
            self.name, self.category, self.start, duration,
            None, None, None, self.caller.take(), None,
        ));
    }
}

/// Create a trace guard for an operation.
pub(crate) fn trace_op(name: &'static str, category: OpCategory) -> TraceGuard {
    TraceGuard::new(name, category)
}

/// Record a completed operation with all optional metadata (caller, data preview).
pub(crate) fn record_op_full(
    name: &str, category: OpCategory, start: Instant, duration: Duration,
    input_shapes: Vec<Vec<usize>>, output_shape: Vec<usize>,
    caller: Option<String>, data_preview: Option<Vec<f64>>,
) {
    let input_shapes = if input_shapes.is_empty() { None } else { Some(input_shapes) };
    with_tracer(|t| t.record_full(
        name, category, start, duration, None, input_shapes, Some(output_shape),
        caller, data_preview,
    ));
}

/// Record a fusion optimization execution.
#[cfg(feature = "fusion")]
pub(crate) fn record_fusion_execute(kind: FusionKind, num_fused_ops: usize, fused_ops: Vec<FusedOpInfo>, start: Instant, duration: Duration, caller: Option<String>, fusion_inputs: Vec<FusionTensorData>, fusion_outputs: Vec<FusionTensorData>, kernel_sources: Vec<String>) {
    with_tracer(|t| t.record_fusion(kind, num_fused_ops, fused_ops, start, duration, caller, fusion_inputs, fusion_outputs, kernel_sources));
}

// ---------------------------------------------------------------------------
// trace-caller: backtrace-based caller location capture
// ---------------------------------------------------------------------------

/// Capture the caller location from a backtrace (returns None when feature is off).
pub(crate) fn maybe_capture_caller() -> Option<String> {
    #[cfg(feature = "trace-caller")]
    {
        let bt = std::backtrace::Backtrace::force_capture();
        let s = format!("{bt}");
        parse_user_frame(&s)
    }
    #[cfg(not(feature = "trace-caller"))]
    {
        None
    }
}

/// Parse a `std::backtrace::Backtrace` string to find the first user-code frame.
///
/// Backtrace format varies by platform. Common patterns:
/// ```text
///    N: crate::module::function            (Linux/Mac)
///    N: 0x7ff... - crate::module::function (Windows MSVC)
///              at /path/to/file.rs:42:5
/// ```
///
/// Strategy: check the **function name** first (most reliable), fall back to
/// **file path** when the function name is unresolved (just an address).
#[cfg(feature = "trace-caller")]
fn parse_user_frame(bt_str: &str) -> Option<String> {
    // None = unknown (no function name yet, or unresolved address)
    // Some(true) = internal frame, Some(false) = user frame
    let mut current_frame_internal: Option<bool> = None;

    for line in bt_str.lines() {
        let trimmed = line.trim();

        // Function-name line: "N: ..." where N is all digits
        if let Some(colon_pos) = trimmed.find(": ") {
            let before = &trimmed[..colon_pos];
            if before.chars().all(|c| c.is_ascii_digit()) {
                let after = &trimmed[colon_pos + 2..];
                // Strip optional address prefix: "0x7ff12345 - real::name"
                let func_name = after
                    .find(" - ")
                    .map(|p| &after[p + 3..])
                    .unwrap_or(after);

                if func_name.starts_with("0x") || !func_name.contains("::") {
                    // Unresolved symbol — we'll fall back to path check
                    current_frame_internal = None;
                } else {
                    current_frame_internal = Some(is_internal_function(func_name));
                }
            }
        }
        // Location line: "at path:line"
        else if let Some(location) = trimmed.strip_prefix("at ") {
            let is_internal = current_frame_internal
                .unwrap_or_else(|| is_internal_path(location));
            if !is_internal {
                return Some(location.trim().to_string());
            }
            current_frame_internal = None;
        }
    }
    None
}

/// Check function name for known internal crate prefixes.
///
/// Handles MSVC name mangling: `enum2$<burn_fusion::...>`, `impl$0::method`,
/// `tuple$<...>`, etc.  We strip any leading MSVC wrapper + `<` to reach the
/// actual crate path, then check prefixes.
#[cfg(feature = "trace-caller")]
fn is_internal_function(func: &str) -> bool {
    // Strip MSVC wrappers like "enum2$<", "tuple$<", "ref$<", etc.
    // Also handles plain "<" for impl blocks like "<burn_fusion::Foo as Trait>::method".
    let name = func
        .find('$')
        .and_then(|dollar| {
            // e.g. "enum2$<burn_fusion::...>" — skip past "$<"
            func.get(dollar + 1..).and_then(|rest| rest.strip_prefix('<'))
        })
        .or_else(|| func.strip_prefix('<'))
        .unwrap_or(func);

    name.starts_with("burn_")    // burn_tensor, burn_fusion, burn_cubecl_fusion, burn_tracing_backend, …
        || name.starts_with("cubecl")  // cubecl, cubecl_runtime, cubecl_wgpu, …
        || name.starts_with("std::")
        || name.starts_with("core::")
        || name.starts_with("futures::")
        || name.starts_with("futures_lite::")
        || name.starts_with("alloc::")
        || name.starts_with("tokio::")
        || name.starts_with("wgpu")
}

/// Fallback: check file path when function name is unavailable/unresolved.
#[cfg(feature = "trace-caller")]
fn is_internal_path(loc: &str) -> bool {
    let loc = loc.replace('\\', "/");
    loc.contains("/burn-")
        || loc.contains("/burn/crates/")
        || loc.contains("/cubecl/")
        || loc.contains("/rustc/")
        || loc.contains("/.cargo/")
}

// ---------------------------------------------------------------------------
// trace-data: tensor data preview capture
// ---------------------------------------------------------------------------

/// Maximum number of tensor values to capture per operation.
#[cfg(feature = "trace-data")]
static DATA_CAPTURE_LIMIT: AtomicUsize = AtomicUsize::new(64);

/// Set the maximum number of tensor values captured per operation (default: 64).
/// Set to 0 to disable data capture while keeping the feature compiled in.
#[cfg(feature = "trace-data")]
pub fn set_data_capture_limit(limit: usize) {
    DATA_CAPTURE_LIMIT.store(limit, Ordering::Relaxed);
}

#[cfg(feature = "trace-data")]
pub(crate) fn data_capture_limit() -> usize {
    DATA_CAPTURE_LIMIT.load(Ordering::Relaxed)
}

/// A minimal `block_on` that resolves a future synchronously using a condvar waker.
fn block_on_future<F: std::future::Future>(f: F) -> F::Output {
    use std::pin::pin;
    use std::sync::{Arc, Condvar, Mutex as StdMutex};
    use std::task::{Context, Poll, Wake, Waker};

    struct CondWake(StdMutex<bool>, Condvar);
    impl Wake for CondWake {
        fn wake(self: Arc<Self>) {
            *self.0.lock().unwrap() = true;
            self.1.notify_one();
        }
    }

    let shared = Arc::new(CondWake(StdMutex::new(false), Condvar::new()));
    let waker = Waker::from(shared.clone());
    let mut cx = Context::from_waker(&waker);
    let mut f = pin!(f);

    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => {
                let mut ready = shared.0.lock().unwrap();
                while !*ready {
                    ready = shared.1.wait(ready).unwrap();
                }
                *ready = false;
            }
        }
    }
}

/// Convert `TensorData` to a `Vec<f64>`, taking at most `limit` values.
fn extract_preview(data: &burn_backend::TensorData, limit: usize) -> Vec<f64> {
    data.iter::<f64>().take(limit).collect()
}

/// Capture a data preview from a float tensor (syncs the device).
pub(crate) fn maybe_capture_float_data<B: burn_backend::Backend>(
    _tensor: &burn_backend::tensor::FloatTensor<B>,
) -> Option<Vec<f64>> {
    #[cfg(feature = "trace-data")]
    {
        let limit = data_capture_limit();
        if limit == 0 {
            return None;
        }
        let device = B::float_device(_tensor);
        let _ = B::sync(&device);
        let clone = _tensor.clone();
        block_on_future(B::float_into_data(clone))
            .ok()
            .map(|data| extract_preview(&data, limit))
    }
    #[cfg(not(feature = "trace-data"))]
    {
        None
    }
}

/// Capture a data preview from an int tensor (syncs the device).
pub(crate) fn maybe_capture_int_data<B: burn_backend::Backend>(
    _tensor: &burn_backend::tensor::IntTensor<B>,
) -> Option<Vec<f64>> {
    #[cfg(feature = "trace-data")]
    {
        let limit = data_capture_limit();
        if limit == 0 {
            return None;
        }
        let device = B::int_device(_tensor);
        let _ = B::sync(&device);
        let clone = _tensor.clone();
        block_on_future(B::int_into_data(clone))
            .ok()
            .map(|data| extract_preview(&data, limit))
    }
    #[cfg(not(feature = "trace-data"))]
    {
        None
    }
}

/// Capture a data preview from a bool tensor (syncs the device).
pub(crate) fn maybe_capture_bool_data<B: burn_backend::Backend>(
    _tensor: &burn_backend::tensor::BoolTensor<B>,
) -> Option<Vec<f64>> {
    #[cfg(feature = "trace-data")]
    {
        let limit = data_capture_limit();
        if limit == 0 {
            return None;
        }
        let device = B::bool_device(_tensor);
        let _ = B::sync(&device);
        let clone = _tensor.clone();
        block_on_future(B::bool_into_data(clone))
            .ok()
            .map(|data| extract_preview(&data, limit))
    }
    #[cfg(not(feature = "trace-data"))]
    {
        None
    }
}

/// Capture a data preview from a quantized tensor (dequantizes, then syncs).
pub(crate) fn maybe_capture_quantized_data<B: burn_backend::Backend>(
    _tensor: &burn_backend::tensor::QuantizedTensor<B>,
) -> Option<Vec<f64>> {
    #[cfg(feature = "trace-data")]
    {
        let limit = data_capture_limit();
        if limit == 0 {
            return None;
        }
        let dequantized = B::dequantize(_tensor.clone());
        let device = B::float_device(&dequantized);
        let _ = B::sync(&device);
        block_on_future(B::float_into_data(dequantized))
            .ok()
            .map(|data| extract_preview(&data, limit))
    }
    #[cfg(not(feature = "trace-data"))]
    {
        None
    }
}

/// Unconditionally capture data from a float tensor (syncs the device).
/// `limit`: `Some(n)` captures at most `n` values, `None` captures all.
fn force_capture_float_data<B: burn_backend::Backend>(
    tensor: &burn_backend::tensor::FloatTensor<B>,
    limit: Option<usize>,
) -> Option<Vec<f64>> {
    let device = B::float_device(tensor);
    let _ = B::sync(&device);
    block_on_future(B::float_into_data(tensor.clone()))
        .ok()
        .map(|data| extract_preview(&data, limit.unwrap_or(usize::MAX)))
}

/// Unconditionally capture data from an int tensor (syncs the device).
/// `limit`: `Some(n)` captures at most `n` values, `None` captures all.
fn force_capture_int_data<B: burn_backend::Backend>(
    tensor: &burn_backend::tensor::IntTensor<B>,
    limit: Option<usize>,
) -> Option<Vec<f64>> {
    let device = B::int_device(tensor);
    let _ = B::sync(&device);
    block_on_future(B::int_into_data(tensor.clone()))
        .ok()
        .map(|data| extract_preview(&data, limit.unwrap_or(usize::MAX)))
}

/// Unconditionally capture data from a bool tensor (syncs the device).
/// `limit`: `Some(n)` captures at most `n` values, `None` captures all.
fn force_capture_bool_data<B: burn_backend::Backend>(
    tensor: &burn_backend::tensor::BoolTensor<B>,
    limit: Option<usize>,
) -> Option<Vec<f64>> {
    let device = B::bool_device(tensor);
    let _ = B::sync(&device);
    block_on_future(B::bool_into_data(tensor.clone()))
        .ok()
        .map(|data| extract_preview(&data, limit.unwrap_or(usize::MAX)))
}

// ---------------------------------------------------------------------------
// Public marker/span API
// ---------------------------------------------------------------------------

/// Builder for constructing user markers and spans with optional metadata.
pub struct MarkerBuilder {
    name: String,
    debug_data: Option<String>,
    binary_data: Option<Vec<u8>>,
    tensors: Vec<FusionTensorData>,
}

impl MarkerBuilder {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            debug_data: None,
            binary_data: None,
            tensors: Vec::new(),
        }
    }

    /// Attach a debug string to this marker/span.
    pub fn debug(mut self, data: impl Into<String>) -> Self {
        self.debug_data = Some(data.into());
        self
    }

    /// Attach binary data to this marker/span.
    pub fn binary(mut self, data: Vec<u8>) -> Self {
        self.binary_data = Some(data);
        self
    }

    /// Attach raw tensor data (shape + values) to this marker/span.
    /// Can be called multiple times to attach multiple tensors.
    pub fn tensor_raw(mut self, shape: Vec<usize>, data: Vec<f64>) -> Self {
        self.tensors.push(FusionTensorData {
            shape,
            dtype: String::new(),
            data,
            tensor_id: None,
        });
        self
    }

    /// Attach a float tensor primitive to this marker/span.
    /// Reads shape and data (syncs the device). Can be called multiple times.
    /// Only captures data when the `trace-data` feature is enabled.
    ///
    /// For a high-level `Tensor<B, D>`, call `.tensor(&t.clone().into_primitive().tensor())`.
    pub fn tensor<B: burn_backend::Backend>(
        mut self,
        tensor: &burn_backend::tensor::FloatTensor<B>,
    ) -> Self {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        let data = maybe_capture_float_data::<B>(tensor);
        if let Some(vals) = data {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
        self
    }

    /// Attach an int tensor primitive to this marker/span.
    /// Reads shape and data (syncs the device). Can be called multiple times.
    /// Only captures data when the `trace-data` feature is enabled.
    ///
    /// For a high-level `Tensor<B, D, Int>`, call `.tensor_int(&t.clone().into_primitive().tensor())`.
    pub fn tensor_int<B: burn_backend::Backend>(
        mut self,
        tensor: &burn_backend::tensor::IntTensor<B>,
    ) -> Self {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        let data = maybe_capture_int_data::<B>(tensor);
        if let Some(vals) = data {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
        self
    }

    /// Attach a bool tensor primitive to this marker/span.
    /// Reads shape and data (syncs the device). Can be called multiple times.
    /// Only captures data when the `trace-data` feature is enabled.
    ///
    /// For a high-level `Tensor<B, D, Bool>`, call `.tensor_bool(&t.clone().into_primitive().tensor())`.
    pub fn tensor_bool<B: burn_backend::Backend>(
        mut self,
        tensor: &burn_backend::tensor::BoolTensor<B>,
    ) -> Self {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        let data = maybe_capture_bool_data::<B>(tensor);
        if let Some(vals) = data {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
        self
    }

    /// Attach a float tensor and **always** capture its data, regardless of
    /// the `trace-data` feature flag. `capture_limit`: `Some(n)` captures at
    /// most `n` values, `None` captures all. Can be called multiple times.
    pub fn tensor_with_data<B: burn_backend::Backend>(
        mut self,
        tensor: &burn_backend::tensor::FloatTensor<B>,
        capture_limit: Option<usize>,
    ) -> Self {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        if let Some(vals) = force_capture_float_data::<B>(tensor, capture_limit) {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
        self
    }

    /// Attach an int tensor and **always** capture its data, regardless of
    /// the `trace-data` feature flag. `capture_limit`: `Some(n)` captures at
    /// most `n` values, `None` captures all. Can be called multiple times.
    pub fn tensor_int_with_data<B: burn_backend::Backend>(
        mut self,
        tensor: &burn_backend::tensor::IntTensor<B>,
        capture_limit: Option<usize>,
    ) -> Self {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        if let Some(vals) = force_capture_int_data::<B>(tensor, capture_limit) {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
        self
    }

    /// Attach a bool tensor and **always** capture its data, regardless of
    /// the `trace-data` feature flag. `capture_limit`: `Some(n)` captures at
    /// most `n` values, `None` captures all. Can be called multiple times.
    pub fn tensor_bool_with_data<B: burn_backend::Backend>(
        mut self,
        tensor: &burn_backend::tensor::BoolTensor<B>,
        capture_limit: Option<usize>,
    ) -> Self {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        if let Some(vals) = force_capture_bool_data::<B>(tensor, capture_limit) {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
        self
    }

    /// Emit an instant marker event (vertical line on the timeline).
    pub fn emit(self) {
        let now = Instant::now();
        let caller = maybe_capture_caller();
        with_tracer(|t| t.record_marker(
            &self.name, now, Duration::ZERO, false,
            self.debug_data, self.binary_data,
            self.tensors, caller,
        ));
    }

    /// Start a span. Returns a guard that records the span when dropped.
    pub fn span(self) -> SpanGuard {
        SpanGuard {
            name: self.name,
            start: Instant::now(),
            caller: maybe_capture_caller(),
            debug_data: self.debug_data,
            binary_data: self.binary_data,
            tensors: self.tensors,
        }
    }
}

/// RAII guard for a user span. The span is recorded when this guard is dropped.
pub struct SpanGuard {
    name: String,
    start: Instant,
    caller: Option<String>,
    debug_data: Option<String>,
    binary_data: Option<Vec<u8>>,
    tensors: Vec<FusionTensorData>,
}

impl SpanGuard {
    /// Attach raw tensor data after the span has started.
    /// Can be called multiple times to attach multiple tensors.
    pub fn add_tensor_raw(&mut self, shape: Vec<usize>, data: Vec<f64>) {
        self.tensors.push(FusionTensorData {
            shape,
            dtype: String::new(),
            data,
            tensor_id: None,
        });
    }

    /// Attach a float tensor primitive after the span has started.
    /// Can be called multiple times.
    pub fn add_tensor<B: burn_backend::Backend>(
        &mut self,
        tensor: &burn_backend::tensor::FloatTensor<B>,
    ) {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        let data = maybe_capture_float_data::<B>(tensor);
        if let Some(vals) = data {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
    }

    /// Attach an int tensor primitive after the span has started.
    /// Can be called multiple times.
    pub fn add_tensor_int<B: burn_backend::Backend>(
        &mut self,
        tensor: &burn_backend::tensor::IntTensor<B>,
    ) {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        let data = maybe_capture_int_data::<B>(tensor);
        if let Some(vals) = data {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
    }

    /// Attach a bool tensor primitive after the span has started.
    /// Can be called multiple times.
    pub fn add_tensor_bool<B: burn_backend::Backend>(
        &mut self,
        tensor: &burn_backend::tensor::BoolTensor<B>,
    ) {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        let data = maybe_capture_bool_data::<B>(tensor);
        if let Some(vals) = data {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
    }

    /// Attach a float tensor after the span has started, **always** capturing
    /// data regardless of the `trace-data` feature flag. `capture_limit`:
    /// `Some(n)` captures at most `n` values, `None` captures all.
    pub fn add_tensor_with_data<B: burn_backend::Backend>(
        &mut self,
        tensor: &burn_backend::tensor::FloatTensor<B>,
        capture_limit: Option<usize>,
    ) {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        if let Some(vals) = force_capture_float_data::<B>(tensor, capture_limit) {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
    }

    /// Attach an int tensor after the span has started, **always** capturing
    /// data regardless of the `trace-data` feature flag. `capture_limit`:
    /// `Some(n)` captures at most `n` values, `None` captures all.
    pub fn add_tensor_int_with_data<B: burn_backend::Backend>(
        &mut self,
        tensor: &burn_backend::tensor::IntTensor<B>,
        capture_limit: Option<usize>,
    ) {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        if let Some(vals) = force_capture_int_data::<B>(tensor, capture_limit) {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
    }

    /// Attach a bool tensor after the span has started, **always** capturing
    /// data regardless of the `trace-data` feature flag. `capture_limit`:
    /// `Some(n)` captures at most `n` values, `None` captures all.
    pub fn add_tensor_bool_with_data<B: burn_backend::Backend>(
        &mut self,
        tensor: &burn_backend::tensor::BoolTensor<B>,
        capture_limit: Option<usize>,
    ) {
        let shape: Vec<usize> = burn_backend::TensorMetadata::shape(tensor).iter().copied().collect();
        if let Some(vals) = force_capture_bool_data::<B>(tensor, capture_limit) {
            let dtype = format!("{:?}", burn_backend::TensorMetadata::dtype(tensor));
            self.tensors.push(FusionTensorData { shape, dtype, data: vals, tensor_id: None });
        }
    }

    /// Attach debug data after the span has started.
    pub fn set_debug(&mut self, data: impl Into<String>) {
        self.debug_data = Some(data.into());
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        with_tracer(|t| t.record_marker(
            &self.name, self.start, duration, true,
            self.debug_data.take(), self.binary_data.take(),
            std::mem::take(&mut self.tensors),
            self.caller.take(),
        ));
    }
}

/// Create a marker builder. Use `.emit()` for an instant marker or `.span()` for a span.
///
/// # Examples
///
/// ```ignore
/// use burn_tracing_backend::marker;
///
/// // Instant marker
/// marker("checkpoint").emit();
///
/// // Marker with debug data
/// marker("loss").debug("epoch 5, loss=0.032").emit();
///
/// // Marker with tensor data (pass primitive via .into_primitive().tensor())
/// marker("output")
///     .tensor::<B>(&result.clone().into_primitive().tensor())
///     .emit();
///
/// // Multiple tensors on one marker
/// marker("layer_io")
///     .tensor::<B>(&input.clone().into_primitive().tensor())
///     .tensor::<B>(&output.clone().into_primitive().tensor())
///     .emit();
///
/// // Span (timed region)
/// let _guard = marker("forward_pass").span();
/// // ... code runs here ...
/// // span recorded when _guard drops
///
/// // Span with tensors attached after computation
/// let mut span = marker("inference").span();
/// // ... compute result ...
/// span.add_tensor::<B>(&result.clone().into_primitive().tensor());
/// drop(span);
/// ```
pub fn marker(name: impl Into<String>) -> MarkerBuilder {
    MarkerBuilder::new(name)
}
