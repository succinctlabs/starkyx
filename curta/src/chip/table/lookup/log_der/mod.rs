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
    pub(crate) challenge: CubicRegister,
    pub(crate) table: Vec<T>,
    pub(crate) multiplicities: ArrayRegister<ElementRegister>,
    pub(crate) multiplicities_table_log: ArrayRegister<CubicRegister>,
    pub(crate) table_accumulator: CubicRegister,
    pub(crate) digest: CubicRegister,
    _marker: core::marker::PhantomData<(F, E)>,
}

/// Currently, only supports an even number of values
#[derive(Debug, Clone)]
pub struct LogLookupValues<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    pub(crate) challenge: CubicRegister,
    pub(crate) values: Vec<T>,
    pub(crate) row_accumulators: ArrayRegister<CubicRegister>,
    pub(crate) log_lookup_accumulator: CubicRegister,
    pub digest: CubicRegister,
    _marker: core::marker::PhantomData<(F, E)>,
}

/// Currently, only supports an even number of values
#[derive(Debug, Clone)]
pub struct LogLookup<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    pub(crate) challenge: CubicRegister,
    pub(crate) table_data: LookupTable<T, F, E>,
    pub(crate) values_data: LogLookupValues<T, F, E>,
    pub(crate) table_index: Option<fn(T::Value<F>) -> usize>,
    _marker: core::marker::PhantomData<(F, E)>,
}

// LogLookUp Memory allocation
impl<L: AirParameters> AirBuilder<L> {
    pub fn lookup_table(
        &mut self,
        challenge: &CubicRegister,
        table: &ElementRegister,
    ) -> LookupTable<ElementRegister, L::Field, L::CubicParams> {
        let multiplicity = self.alloc_array::<ElementRegister>(1);
        let multiplicities_table_log = self.alloc_array_extended::<CubicRegister>(1);
        let table_accumulator = self.alloc_extended::<CubicRegister>();
        LookupTable {
            challenge: *challenge,
            table: vec![*table],
            multiplicities: multiplicity,
            multiplicities_table_log,
            table_accumulator,
            digest: table_accumulator,
            _marker: PhantomData,
        }
    }

    pub fn lookup_values(
        &mut self,
        challenge: &CubicRegister,
        values: &[ElementRegister],
    ) -> LogLookupValues<ElementRegister, L::Field, L::CubicParams> {
        assert_eq!(values.len() % 2, 0, "Only even number of values supported");
        let row_accumulators = self.alloc_array_extended::<CubicRegister>(values.len() / 2);
        let log_lookup_accumulator = self.alloc_extended::<CubicRegister>();

        LogLookupValues {
            challenge: *challenge,
            values: values.to_vec(),
            row_accumulators,
            log_lookup_accumulator,
            digest: log_lookup_accumulator,
            _marker: PhantomData,
        }
    }

    pub fn lookup_log_derivative(
        &mut self,
        table: &ElementRegister,
        values: &[ElementRegister],
        table_index: fn(L::Field) -> usize,
    ) {
        // Allocate memory for the lookup
        let challenge = self.alloc_challenge::<CubicRegister>();

        let table_data = self.lookup_table(&challenge, table);
        let value_data = self.lookup_values(&challenge, values);

        let lookup_data = Lookup::LogDerivative(LogLookup {
            challenge,
            table_data,
            values_data: value_data,
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
