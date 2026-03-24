use burn_backend::{Backend, ops::TransactionOps};

use crate::backend::Profiler;

// TransactionOps default implementation calls Self::float_into_data, Self::int_into_data, etc.
// which are our traced methods. So the default gives us tracing for free.
impl<B: Backend> TransactionOps<Self> for Profiler<B> {}
