/// Helper macro: extract shape from an argument if it's a tensor, skip otherwise.
/// Uses autoref specialization: `(&tensor).maybe_shape()` resolves to TensorMetadata impl,
/// while `(&&non_tensor).maybe_shape()` resolves to the fallback returning None.
macro_rules! maybe_shape {
    ($arg:ident) => {{
        #[allow(unused_imports)]
        use $crate::trace::MaybeShapeFallback;
        // ShapeProbe(&$arg) = ShapeProbe<&T>.
        // Tensor T: inherent maybe_shape() on ShapeProbe<&T> wins → Some(shape).
        // Non-tensor T: no inherent method → falls back to MaybeShapeFallback trait → None.
        $crate::trace::ShapeProbe(&$arg).maybe_shape()
    }};
}

/// Delegation macro for simple operations that return a single tensor.
/// Captures input shapes (from tensor args) and output shape via TensorMetadata.
/// Also captures caller location (trace-caller) and data preview (trace-data).
///
/// Pattern-matches on the return type to choose the right data capture function:
/// - `-> FloatTensor<B>` → `maybe_capture_float_data`
/// - `-> IntTensor<B>`   → `maybe_capture_int_data`
/// - `-> BoolTensor<B>`  → `maybe_capture_bool_data`
/// - anything else       → no data capture
macro_rules! delegate {
    // Float return type
    ($cat:expr, fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> FloatTensor < B >) => {
        delegate!(@impl $cat, $name, ($($arg: $ty),*) -> FloatTensor<B>,
            $crate::trace::maybe_capture_float_data::<B>);
    };
    // Int return type
    ($cat:expr, fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> IntTensor < B >) => {
        delegate!(@impl $cat, $name, ($($arg: $ty),*) -> IntTensor<B>,
            $crate::trace::maybe_capture_int_data::<B>);
    };
    // Bool return type
    ($cat:expr, fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> BoolTensor < B >) => {
        delegate!(@impl $cat, $name, ($($arg: $ty),*) -> BoolTensor<B>,
            $crate::trace::maybe_capture_bool_data::<B>);
    };
    // Fallback — unknown return type, no data capture
    ($cat:expr, fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty) => {
        delegate!(@impl_no_data $cat, $name, ($($arg: $ty),*) -> $ret);
    };

    // ---- internal implementation arms ----

    (@impl $cat:expr, $name:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty, $capture_fn:path) => {
        fn $name($($arg: $ty),*) -> $ret {
            let _input_shapes: Vec<Vec<usize>> = [$(maybe_shape!($arg)),*]
                .into_iter()
                .flatten()
                .collect();
            let _caller = $crate::trace::maybe_capture_caller();
            let _start = std::time::Instant::now();
            let _result = B::$name($($arg),*);
            let _duration = _start.elapsed();
            let _out_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&_result).iter().copied().collect();
            let _data = $capture_fn(&_result);
            $crate::trace::record_op_full(stringify!($name), $cat, _start, _duration, _input_shapes, _out_shape, _caller, _data);
            _result
        }
    };
    (@impl_no_data $cat:expr, $name:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name($($arg: $ty),*) -> $ret {
            let _input_shapes: Vec<Vec<usize>> = [$(maybe_shape!($arg)),*]
                .into_iter()
                .flatten()
                .collect();
            let _caller = $crate::trace::maybe_capture_caller();
            let _start = std::time::Instant::now();
            let _result = B::$name($($arg),*);
            let _duration = _start.elapsed();
            let _out_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&_result).iter().copied().collect();
            $crate::trace::record_op_full(stringify!($name), $cat, _start, _duration, _input_shapes, _out_shape, _caller, None);
            _result
        }
    };
}

/// Helper macro for manual ops that return a single tensor.
/// Usage: `trace_with_shape!(Category, "name", expr)`
/// Note: input shapes must be captured manually before the call if needed.
/// Captures caller location but not data preview by default.
/// Use the `data:` variant for data capture: `trace_with_shape!(Cat, "name", [inputs], expr, data: capture_fn)`
macro_rules! trace_with_shape {
    ($cat:expr, $name:expr, $body:expr) => {{
        let _caller = $crate::trace::maybe_capture_caller();
        let _start = std::time::Instant::now();
        let _result = $body;
        let _duration = _start.elapsed();
        let _out_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&_result).iter().copied().collect();
        $crate::trace::record_op_full($name, $cat, _start, _duration, vec![], _out_shape, _caller, None);
        _result
    }};
    ($cat:expr, $name:expr, [$($input:expr),*], $body:expr) => {{
        let _input_shapes: Vec<Vec<usize>> = vec![$($input),*];
        let _caller = $crate::trace::maybe_capture_caller();
        let _start = std::time::Instant::now();
        let _result = $body;
        let _duration = _start.elapsed();
        let _out_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&_result).iter().copied().collect();
        $crate::trace::record_op_full($name, $cat, _start, _duration, _input_shapes, _out_shape, _caller, None);
        _result
    }};
    // With data capture (no input shapes)
    ($cat:expr, $name:expr, $body:expr, data: $capture_fn:path) => {{
        let _caller = $crate::trace::maybe_capture_caller();
        let _start = std::time::Instant::now();
        let _result = $body;
        let _duration = _start.elapsed();
        let _out_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&_result).iter().copied().collect();
        let _data = $capture_fn(&_result);
        $crate::trace::record_op_full($name, $cat, _start, _duration, vec![], _out_shape, _caller, _data);
        _result
    }};
    // With data capture (with input shapes)
    ($cat:expr, $name:expr, [$($input:expr),*], $body:expr, data: $capture_fn:path) => {{
        let _input_shapes: Vec<Vec<usize>> = vec![$($input),*];
        let _caller = $crate::trace::maybe_capture_caller();
        let _start = std::time::Instant::now();
        let _result = $body;
        let _duration = _start.elapsed();
        let _out_shape: Vec<usize> = burn_backend::TensorMetadata::shape(&_result).iter().copied().collect();
        let _data = $capture_fn(&_result);
        $crate::trace::record_op_full($name, $cat, _start, _duration, _input_shapes, _out_shape, _caller, _data);
        _result
    }};
}

pub(crate) mod float;
pub(crate) mod int;
pub(crate) mod bool_ops;
pub(crate) mod module;
pub(crate) mod activation;
pub(crate) mod qtensor;
pub(crate) mod transaction;
