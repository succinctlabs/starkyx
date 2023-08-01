//! This module implements a lookup argument based on the logarithmic derivative as in
//! https://eprint.iacr.org/2022/1530.pdf
//!

use core::marker::PhantomData;

use crate::chip::builder::AirBuilder;
use crate::chip::constraint::Constraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::table::lookup::Lookup;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub mod constraint;
pub mod trace;

#[derive(Debug, Clone)]
pub struct LookupTable<T: Register, F: Field, E: CubicParameters<F>> {
    pub(crate) table: Vec<T>,
    pub(crate) multiplicities: ArrayRegister<ElementRegister>,
    _marker: core::marker::PhantomData<(F, E)>,
}

/// Currently, only supports an even number of values
#[derive(Debug, Clone)]
pub struct LogLookup<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    pub(crate) challenge: CubicRegister,
    pub(crate) table_data: LookupTable<T, F, E>,
    pub(crate) values: Vec<T>,
    pub(crate) multiplicity_table_log: CubicRegister,
    pub(crate) row_accumulators: ArrayRegister<CubicRegister>,
    pub(crate) log_lookup_accumulator: CubicRegister,
    table_index: Option<fn(T::Value<F>) -> usize>,
    _marker: core::marker::PhantomData<(F, E)>,
}

// LogLookUp Memory allocation
impl<L: AirParameters> AirBuilder<L> {
    pub fn lookup_log_derivative(
        &mut self,
        table: &ElementRegister,
        values: &[ElementRegister],
        table_index: fn(L::Field) -> usize,
    ) {
        // Allocate memory for the lookup
        let challenge = self.alloc_challenge::<CubicRegister>();
        let multiplicities = self.alloc_array_extended::<ElementRegister>(1);
        let multiplicity_table_log = self.alloc_extended::<CubicRegister>();
        let row_accumulators = self.alloc_array_extended::<CubicRegister>(values.len() / 2);
        let log_lookup_accumulator = self.alloc_extended::<CubicRegister>();
        let table = ArrayRegister::from_element(*table)
            .into_iter()
            .collect::<Vec<_>>();

        let lookup_data = Lookup::LogDerivative(LogLookup {
            challenge,
            table_data: LookupTable {
                table,
                multiplicities,
                _marker: PhantomData,
            },
            values: values.to_vec(),
            multiplicity_table_log,
            row_accumulators,
            log_lookup_accumulator,
            table_index: Some(table_index),
            _marker: core::marker::PhantomData,
        });

        // Add the lookup constraints
        self.constraints
            .push(Constraint::lookup(lookup_data.clone()));

        // Add the lookup to the list of lookups
        self.lookup_data.push(lookup_data);
    }
}
