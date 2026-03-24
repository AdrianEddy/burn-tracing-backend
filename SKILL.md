# Burn Tracing Backend — Trace Data Format Reference

This document describes the data format of `trace_data.js` files produced by the [burn-tracing-backend](https://github.com/AdrianEddy/burn-tracing-backend) profiler for the [Burn](https://burn.dev) deep learning framework. Use this to write targeted Python scripts that parse trace data for ML model optimization, profiling, and performance analysis.

## File Format

The trace file is a JavaScript file defining a single constant:

```js
const TRACE_DATA = [ ...array of event objects... ];
```

### How to Load in Python

```python
import json

def load_trace(path: str) -> list[dict]:
    """Load trace_data.js into a list of event dicts."""
    text = open(path).read()
    # File format is: const TRACE_DATA = [...];\n
    return json.loads(text[len("const TRACE_DATA = "):].strip().rstrip(";"))

events = load_trace("trace_data.js")
```

## Event Object Schema

Every event in the array is a JSON object. Fields marked **(optional)** are omitted when not applicable (use `.get()` in Python).

### Core Fields (always present)

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Operation name. Examples: `"float_random"`, `"float_matmul"`, `"float_add"`, `"sync"`, `"fusion::elementwise"`, `"fusion::matmul"`, `"my_marker"` |
| `category` | `string` | One of: `"float"`, `"int"`, `"bool"`, `"module"`, `"activation"`, `"quantized"`, `"transaction"`, `"sync"`, `"fusion"`, `"marker"` |
| `start_us` | `float` | Microseconds since trace start (wall-clock offset from first event) |
| `duration_us` | `float` | Duration in microseconds (CPU-side timing) |
| `op_index` | `int` | Sequential index — events are ordered by execution sequence |

### Shape & Memory Fields (optional)

| Field | Type | Description |
|---|---|---|
| `input_shapes` | `list[list[int]]` | Shapes of input tensors, e.g. `[[4, 16], [16, 32]]` for matmul |
| `output_shape` | `list[int]` | Shape of the output tensor, e.g. `[4, 32]` |
| `memory_bytes` | `int` | Estimated memory allocation in bytes for this operation's output |

### Tensor Data Preview Fields (optional, requires `trace-data` feature)

| Field | Type | Description |
|---|---|---|
| `data_preview` | `list[float]` | First N values of the output tensor (default N=64). Used for non-fusion ops |
| `data_shape` | `list[int]` | Shape of the tensor previewed in `data_preview` |

### Caller Location (optional, requires `trace-caller` feature)

| Field | Type | Description |
|---|---|---|
| `caller` | `string` | Source location like `"./examples/sample_inference.rs:15"`. Not available for fused ops |

### GPU Sync Fields (optional)

| Field | Type | Description |
|---|---|---|
| `is_sync` | `bool` | `true` if this op forces GPU synchronization. Appears on explicit `sync()` calls and implicit syncs like `into_data()` / `to_data()`. Sync events are performance-critical — they block until GPU work completes |

### Fusion Fields (optional, only on `category == "fusion"` events)

Fusion events represent multiple operations that the GPU compiler fused into a single kernel launch.

| Field | Type | Description |
|---|---|---|
| `fusion_kind` | `string` | One of: `"elementwise"`, `"matmul"`, `"reduce"`, `"reducebroadcasted"` |
| `num_fused_ops` | `int` | Number of operations fused into this kernel |
| `fused_ops` | `list[FusedOpInfo]` | Details of each individual operation in the fusion block (see below) |
| `fusion_inputs` | `list[FusionTensorData]` | Input tensor data for the fusion block (with `trace-data` feature) |
| `fusion_outputs` | `list[FusionTensorData]` | Output tensor data for the fusion block (with `trace-data` feature) |
| `kernel_sources` | `list[string]` | Compiled GPU kernel source code(s) (with `trace-data` feature). Typically one per fusion, sometimes two for matmul |

#### `FusedOpInfo` Object

Each element of `fused_ops` describes one operation within a fusion block:

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Operation name with category prefix, e.g. `"float::Add"`, `"float::Exp"`, `"module::Conv2d"` |
| `input_shapes` | `list[list[int]]` | Input tensor shapes for this sub-operation |
| `output_shapes` | `list[list[int]]` | Output tensor shapes for this sub-operation |
| `input_ids` | `list[string]` | Tensor IDs of inputs (may be empty). Links tensors across operations within a fusion |
| `output_ids` | `list[string]` | Tensor IDs of outputs (may be empty) |
| `is_fallback` | `bool` or absent | `true` if this op was NOT fused but ran as a separate kernel (fallback). Absent or `null` means it was successfully fused |

#### `FusionTensorData` Object

Each element of `fusion_inputs` / `fusion_outputs`:

| Field | Type | Description |
|---|---|---|
| `shape` | `list[int]` | Tensor shape |
| `dtype` | `string` | Element type, e.g. `"F32"`, `"Bool(U32)"` |
| `data` | `list[float]` | First N values as float64 |
| `tensor_id` | `string` or absent | Links to a specific fused op's input/output ID |

### Fusion Fallback Fields (optional)

| Field | Type | Description |
|---|---|---|
| `is_fusion_fallback` | `bool` | `true` on standalone events that were part of a fusion pattern but fell back to separate kernel execution |

### Marker/Span Fields (optional, only on `category == "marker"` events)

| Field | Type | Description |
|---|---|---|
| `is_span` | `bool` | `true` = timed span (has meaningful duration), `false` = instant marker (duration ~0) |
| `debug_data` | `string` | User-provided debug string, e.g. `"Epoch 5 complete, loss=0.032"` |
| `binary_data` | `string` | Base64-encoded binary data attached by user |

Marker events may also have `fusion_outputs` containing attached tensor data.

## Categories Explained

| Category | What it captures | Typical names |
|---|---|---|
| `float` | Float tensor operations | `float_random`, `float_matmul`, `float_add`, `float_mul`, `float_exp`, `float_reshape`, `float_swap_dims`, `float_zeros`, `float_ones`, `float_into_data` |
| `int` | Integer tensor operations | `int_arange`, `int_add`, `int_reshape` |
| `bool` | Boolean tensor operations | `bool_not`, `bool_and` |
| `module` | Neural network module ops | `module_conv2d`, `module_batch_norm`, `module_embedding` |
| `activation` | Activation functions | `activation_relu`, `activation_gelu`, `activation_sigmoid` |
| `quantized` | Quantized tensor operations | Quantization-related ops |
| `transaction` | Batch tensor operations | Transaction/batch ops |
| `sync` | GPU synchronization points | `sync` (explicit), or regular ops with `is_sync: true` (implicit) |
| `fusion` | Fused GPU kernels | `fusion::elementwise`, `fusion::matmul`, `fusion::reduce`, `fusion::reduce_broadcasted` |
| `marker` | User-placed markers/spans | Any user-defined name |

## Timing Model

- All times are **CPU-side wall-clock** measurements in microseconds
- `start_us` is relative to the first traced event (epoch = 0)
- Events are ordered by `op_index` (execution order), NOT by `start_us`
- GPU operations are dispatched asynchronously — `duration_us` reflects CPU dispatch time, NOT GPU execution time
- **Sync events are the exception**: their `duration_us` includes actual GPU wait time, making them key for identifying GPU bottlenecks
- The total trace wall time is approximately `max(start_us + duration_us)` across all events

## Common Analysis Patterns

### 1. Find the most expensive operations

```python
events.sort(key=lambda e: e['duration_us'], reverse=True)
for e in events[:20]:
    print(f"{e['duration_us']:.0f}us  {e['name']}  {e.get('output_shape', '')}")
```

### 2. Compute fusion ratio

```python
total_ops = sum(1 for e in events if e['category'] not in ('sync', 'marker'))
fused_ops = sum(e.get('num_fused_ops', 0) for e in events if e['category'] == 'fusion')
fusion_events = sum(1 for e in events if e['category'] == 'fusion')
print(f"Fusion ratio: {fused_ops}/{total_ops} ops fused into {fusion_events} kernels")
```

### 3. Find GPU sync bottlenecks

```python
syncs = [e for e in events if e.get('is_sync') or e['category'] == 'sync']
for s in syncs:
    print(f"{s['start_us']:.0f}us  {s['duration_us']:.0f}us  {s['name']}")
```

### 4. Memory allocation timeline

```python
allocs = [(e['start_us'], e['memory_bytes'], e['name'], e.get('output_shape'))
          for e in events if e.get('memory_bytes')]
total_mem = sum(b for _, b, _, _ in allocs)
print(f"Total allocated: {total_mem / 1e6:.1f} MB across {len(allocs)} ops")
```

### 5. Analyze tensor shapes through a model

```python
for e in events:
    if e.get('input_shapes') and e.get('output_shape'):
        print(f"{e['name']:30s}  {e['input_shapes']} -> {e['output_shape']}")
```

### 6. Extract operations within a user-defined span

```python
span = next(e for e in events if e['category'] == 'marker' and e.get('is_span') and e['name'] == 'forward_pass')
span_start = span['start_us']
span_end = span_start + span['duration_us']
ops_in_span = [e for e in events if e['category'] != 'marker'
               and e['start_us'] >= span_start and e['start_us'] <= span_end]
```

### 7. Identify fallback ops in fusion blocks

```python
for e in events:
    if e.get('fused_ops'):
        fallbacks = [op for op in e['fused_ops'] if op.get('is_fallback')]
        if fallbacks:
            print(f"Fusion {e['name']} has {len(fallbacks)} fallback ops: "
                  f"{[op['name'] for op in fallbacks]}")
```

### 8. Category-level timing breakdown

```python
from collections import defaultdict
by_cat = defaultdict(lambda: {'count': 0, 'total_us': 0.0})
for e in events:
    by_cat[e['category']]['count'] += 1
    by_cat[e['category']]['total_us'] += e['duration_us']
for cat, stats in sorted(by_cat.items(), key=lambda x: -x[1]['total_us']):
    print(f"{cat:15s}  {stats['count']:5d} ops  {stats['total_us']:12.0f}us")
```

### 9. Matmul shapes analysis (for optimization)

```python
matmuls = [e for e in events if 'matmul' in e['name'].lower()]
for m in matmuls:
    shapes = m.get('input_shapes', [])
    if len(shapes) == 2:
        M, K1 = shapes[0][-2], shapes[0][-1]
        K2, N = shapes[1][-2], shapes[1][-1]
        print(f"Matmul: [{M}x{K1}] @ [{K2}x{N}] -> {m.get('output_shape')}  {m['duration_us']:.0f}us")
```

### 10. Data flow through fusion blocks

```python
for e in events:
    if e.get('fused_ops'):
        print(f"\n=== {e['name']} ({e.get('num_fused_ops', 0)} ops) ===")
        for op in e['fused_ops']:
            fb = " [FALLBACK]" if op.get('is_fallback') else ""
            print(f"  {op['name']:30s} {op['input_shapes']} -> {op['output_shapes']}{fb}")
```

## Rust API Reference

All public items are re-exported from the crate root: `use burn_tracing_backend::*;`

### Backend Decorator

`Profiler<B>` wraps any Burn backend to intercept all tensor operations.

```rust
use burn_tracing_backend::Profiler;

// Without fusion — traces individual ops
type B = Profiler<CubeBackend<WgpuRuntime, f32, i32, u32>>;

// With fusion — also captures fused kernel dispatch
use burn_fusion::Fusion;
type B = Fusion<Profiler<CubeBackend<WgpuRuntime, f32, i32, u32>>>;
```

`Profiler` must be the **inner** layer when using `Fusion` — `Fusion<Profiler<B>>`, not `Profiler<Fusion<B>>`.

### Feature Flags

```toml
[dependencies]
burn-tracing-backend = { git = "...", features = ["fusion", "trace-caller", "trace-data"] }
```

| Feature | Effect |
|---|---|
| `fusion` | Intercepts CubeCL fusion dispatch — records `fusion::*` events with `fused_ops`, `kernel_sources` |
| `trace-caller` | Captures `file:line:col` via backtrace for each op (not available for fused ops) |
| `trace-data` | Captures tensor value previews (first N elements, default 64) and compiled kernel sources |

### Tracing Lifecycle

```rust
use burn_tracing_backend::{start_tracing, finish_tracing, snapshot_events, write_trace};

start_tracing();                         // clear previous data, begin recording
// ... run model ...
let events = snapshot_events();          // peek at events without stopping
let events = finish_tracing();           // stop recording, return all events (deduplicated)

write_trace(&events, "trace_data.js");   // writes trace_data.js + trace_explorer.html
write_trace_js(&events, "trace_data.js");// writes only the JS file
let js_string = generate_data_js(&events); // returns JS string in memory
```

### Data Capture Limit

```rust
use burn_tracing_backend::set_data_capture_limit;
set_data_capture_limit(128); // first 128 values per tensor (default: 64, 0 = disabled)
```

### Markers and Spans

`marker(name)` returns a builder. Call `.emit()` for an instant event or `.span()` for a timed region.

```rust
use burn_tracing_backend::marker;

// Instant markers (vertical line on timeline)
marker("checkpoint").emit();
marker("loss").debug(format!("epoch {epoch}, loss={loss:.4f}")).emit();

// Timed spans (RAII — duration recorded on drop)
{
    let _s = marker("forward_pass").debug("layers 1-3").span();
    // ... operations traced inside this span ...
}

// Per-layer profiling
for (i, layer) in layers.iter().enumerate() {
    let _s = marker(&format!("layer_{i}")).span();
    x = layer.forward(x);
}
```

#### Attaching tensors (requires `trace-data`)

Tensor data appears as `fusion_outputs` on the marker event. Triggers GPU sync — use sparingly.

```rust
// Float tensor on a marker
marker("output")
    .tensor::<B>(&result.clone().into_primitive().tensor())
    .emit();

// Multiple tensors, int, bool variants
marker("layer_io")
    .tensor::<B>(&input.clone().into_primitive().tensor())
    .tensor::<B>(&output.clone().into_primitive().tensor())
    .emit();
marker("indices").tensor_int::<B>(&idx.clone().into_primitive().tensor()).emit();
marker("mask").tensor_bool::<B>(&mask.clone().into_primitive().tensor()).emit();

// Attach tensors to a span AFTER computation
let mut span = marker("inference").span();
let result = model.forward(input);
span.add_tensor::<B>(&result.clone().into_primitive().tensor());
span.set_debug("done");
drop(span);
```

#### Always-on tensor capture (`*_with_data` — no feature flag needed)

`tensor_with_data` / `add_tensor_with_data` variants always capture data, even without `trace-data`.
Second argument is `capture_limit`: `Some(n)` = first n values, `None` = entire tensor.

```rust
// Capture first 128 values regardless of trace-data feature
marker("weights")
    .tensor_with_data::<B>(&w.clone().into_primitive().tensor(), Some(128))
    .emit();

// Capture ALL values (careful with large tensors!)
marker("small_output")
    .tensor_with_data::<B>(&out.clone().into_primitive().tensor(), None)
    .emit();

// Int/bool variants work the same way
marker("ids").tensor_int_with_data::<B>(&ids.clone().into_primitive().tensor(), Some(64)).emit();

// On spans — attach after computation
let mut span = marker("layer").span();
let y = layer.forward(x);
span.add_tensor_with_data::<B>(&y.clone().into_primitive().tensor(), Some(256));
drop(span);
```

#### Binary data and raw tensors

```rust
marker("snapshot").binary(some_bytes).debug("weights v2").emit();
marker("manual").tensor_raw(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).emit();
```

## Size Expectations

For a typical model inference trace:
- Small model (few layers): hundreds of events, ~50-200 KB JS file
- Medium model (ResNet/BERT): thousands of events, ~1-10 MB JS file
- Large model or training loop: tens/hundreds of thousands of events, ~50-500+ MB JS file

The file is a single JSON array on one line. For very large traces, use streaming JSON parsing or line-based extraction rather than loading the entire file into memory.

## Key Insights for Optimization Work

1. **Fusion is king**: Operations that get fused into a single kernel launch are dramatically cheaper than separate kernel launches. Look at fusion ratio and identify ops that fail to fuse (`is_fallback`, `is_fusion_fallback`).

2. **Sync points are walls**: Every `sync` or `is_sync` event blocks the CPU until all queued GPU work completes. Minimizing sync points (especially implicit ones from `into_data`/`to_data`) is often the single biggest optimization.

3. **CPU timing != GPU timing**: Most operation durations reflect CPU dispatch time (microseconds). The real GPU cost is hidden — it's only visible through sync point durations. A matmul might show 10us dispatch but the sync after it takes 5000us.

4. **Memory estimation**: `memory_bytes` is an estimate of output tensor allocation size. Use it for memory pressure analysis, not exact accounting.

5. **Kernel sources**: When available (`kernel_sources` on fusion events), these are the actual compiled GPU shader code. Useful for understanding what the GPU compiler did with fused operations.
