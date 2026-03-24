#[macro_use]
mod ops;

mod backend;
mod html;
pub mod trace;

#[cfg(feature = "fusion")]
pub mod fusion_runtime;

pub use backend::Profiler;
pub use html::{generate_data_js, write_trace, write_trace_js};
pub use trace::{OpCategory, TraceEvent, finish_tracing, marker, snapshot_events, start_tracing};
pub use trace::{MarkerBuilder, SpanGuard};
#[cfg(feature = "fusion")]
pub use trace::FusionKind;
#[cfg(feature = "trace-data")]
pub use trace::set_data_capture_limit;
