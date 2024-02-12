pub mod constraint;
pub mod trace;

use serde::{Deserialize, Serialize};

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::table::log_derivative::entry::LogEntry;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum LookupValues<F, E> {
    Element(LogLookupValues<ElementRegister, F, E>),
    Cubic(LogLookupValues<CubicRegister, F, E>),
}

/// Currently, only supports an even number of values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogLookupValues<T: EvalCubic, F, E> {
    pub(crate) challenge: CubicRegister,
    pub(crate) trace_values: Vec<LogEntry<T>>,
    pub(crate) public_values: Vec<LogEntry<T>>,
    pub(crate) row_accumulators: ArrayRegister<CubicRegister>,
    pub(crate) global_accumulators: ArrayRegister<CubicRegister>,
    pub local_digest: CubicRegister,
    pub global_digest: Option<CubicRegister>,
    pub digest: CubicRegister,
    pub(crate) _marker: core::marker::PhantomData<(F, E)>,
}
