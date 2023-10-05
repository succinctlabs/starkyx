pub mod constraint;
pub mod trace;

use serde::{Deserialize, Serialize};

use crate::chip::AirParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::math::prelude::*;

/// Currently, only supports an even number of values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogLookupValues<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    pub(crate) challenge: CubicRegister,
    pub(crate) trace_values: Vec<T>,
    pub(crate) public_values: Vec<T>,
    pub(crate) row_accumulators: ArrayRegister<CubicRegister>,
    pub(crate) global_accumulators: ArrayRegister<CubicRegister>,
    pub(crate) log_lookup_accumulator: CubicRegister,
    pub local_digest: CubicRegister,
    pub global_digest: Option<CubicRegister>,
    pub digest: CubicRegister,
    pub(crate) _marker: core::marker::PhantomData<(F, E)>,
}


