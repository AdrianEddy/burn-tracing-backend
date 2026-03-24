use crate::trace::TraceEvent;
use std::path::Path;

/// The trace explorer HTML, embedded at compile time.
const TRACE_EXPLORER_HTML: &[u8] = include_bytes!("../trace_explorer.html");

/// Generate the JS data file contents from trace events.
///
/// Produces `const TRACE_DATA = [...];` suitable for inclusion via `<script src="trace_data.js">`.
pub fn generate_data_js(events: &[TraceEvent]) -> String {
    let json = serde_json::to_string(events).unwrap_or_else(|_| "[]".to_string());
    format!("const TRACE_DATA = {};\n", json)
}

/// Write trace data as a JS file that can be loaded by the visualiser.
///
/// The generated file defines `const TRACE_DATA = [...];`.
/// Place it next to `visualiser.html` and serve via localhost.
pub fn write_trace_js(events: &[TraceEvent], path: &str) -> std::io::Result<()> {
    std::fs::write(path, generate_data_js(events))
}

/// Write trace data as `trace_data.js` alongside `trace_explorer.html`.
///
/// Both files are written into the directory of `path`. If `path` points to a
/// directory the files are created directly inside it; otherwise the parent
/// directory of `path` is used (and `path` itself is used for the JS file name,
/// for backwards compatibility).
///
/// The HTML file is the trace explorer UI, statically embedded in this crate.
pub fn write_trace(events: &[TraceEvent], path: &str) -> std::io::Result<()> {
    let path = Path::new(path);
    let (dir, js_path) = if path.is_dir() {
        (path.to_path_buf(), path.join("trace_data.js"))
    } else {
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        (dir, path.to_path_buf())
    };

    std::fs::write(&js_path, generate_data_js(events))?;
    std::fs::write(dir.join("trace_explorer.html"), TRACE_EXPLORER_HTML)?;
    Ok(())
}
