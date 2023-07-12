//! This module implements a lookup argument based on the logarithmic derivative as in
//! https://eprint.iacr.org/2022/1530.pdf
//!

use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::extension::ExtensionRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub mod constraint;
pub mod trace;

#[derive(Debug, Clone)]
pub struct LogLookup<F: Field, E: ExtensionField<F>, const N: usize>
where
    [(); E::D]:,
{
    pub(crate) challenge: ExtensionRegister<{ E::D }>,
    pub(crate) table: ArrayRegister<ElementRegister>,
    pub(crate) values: ArrayRegister<ElementRegister>,
    pub(crate) multiplicities: ArrayRegister<ElementRegister>,
    pub(crate) multiplicity_table_log: ExtensionRegister<{ E::D }>,
    pub(crate) row_accumulators: ArrayRegister<ExtensionRegister<{ E::D }>>,
    pub(crate) log_lookup_accumulator: ExtensionRegister<{ E::D }>,
    _marker: core::marker::PhantomData<(F, E)>,
}

// LogLookUp Memory allocation
impl<L: AirParameters> AirBuilder<L>
where
    [(); L::Challenge::D]:,
{
    pub(crate) fn lookup_log_derivative(
        &mut self,
        table: &ElementRegister,
        values: &ArrayRegister<ElementRegister>,
    ) {
        let challenge = self.alloc_challenge::<ExtensionRegister<{ L::Challenge::D }>>();
        let multiplicities = self.alloc_array::<ElementRegister>(1);
        let multiplicity_table_log = self.alloc::<ExtensionRegister<{ L::Challenge::D }>>();
        let row_accumulators =
            self.alloc_array::<ExtensionRegister<{ L::Challenge::D }>>(values.len() / 2);
        let log_lookup_accumulator = self.alloc::<ExtensionRegister<{ L::Challenge::D }>>();
        let table = ArrayRegister::from_element(*table);

        self.lookup_data
            .push(super::Lookup::LogDerivative(LogLookup {
                challenge,
                table,
                values: *values,
                multiplicities,
                multiplicity_table_log,
                row_accumulators,
                log_lookup_accumulator,
                _marker: core::marker::PhantomData,
            }));
    }

    pub(crate) fn lookup_log_derivative_split_table(
        &mut self,
        table: &ArrayRegister<ElementRegister>,
        values: &ArrayRegister<ElementRegister>,
    ) {
        let challenge = self.alloc_challenge::<ExtensionRegister<{ L::Challenge::D }>>();
        let multiplicities = self.alloc_array::<ElementRegister>(2);
        let multiplicity_table_log = self.alloc::<ExtensionRegister<{ L::Challenge::D }>>();
        let row_accumulators =
            self.alloc_array::<ExtensionRegister<{ L::Challenge::D }>>(values.len() / 2);
        let log_lookup_accumulator = self.alloc::<ExtensionRegister<{ L::Challenge::D }>>();

        self.lookup_data_split_table
            .push(super::Lookup::LogDerivative(LogLookup {
                challenge,
                table: *table,
                values: *values,
                multiplicities,
                multiplicity_table_log,
                row_accumulators,
                log_lookup_accumulator,
                _marker: core::marker::PhantomData,
            }));
    }
}
