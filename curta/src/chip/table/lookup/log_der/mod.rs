//! This module implements a lookup argument based on the logarithmic derivative as in
//! https://eprint.iacr.org/2022/1530.pdf
//!

use crate::chip::builder::AirBuilder;
use crate::chip::constraint::Constraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::extension::ExtensionRegister;
use crate::chip::table::lookup::Lookup;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub mod constraint;
pub mod trace;

/// Only supports degree 3 extensions
#[derive(Debug, Clone)]
pub struct LogLookup<F: Field, E: CubicParameters<F>, const N: usize> {
    pub(crate) challenge: ExtensionRegister<3>,
    pub(crate) table: ArrayRegister<ElementRegister>,
    pub(crate) values: ArrayRegister<ElementRegister>,
    pub(crate) multiplicities: ArrayRegister<ElementRegister>,
    pub(crate) multiplicity_table_log: ExtensionRegister<3>,
    pub(crate) row_accumulators: ArrayRegister<ExtensionRegister<3>>,
    pub(crate) log_lookup_accumulator: ExtensionRegister<3>,
    table_index: fn(F) -> usize,
    _marker: core::marker::PhantomData<(F, E)>,
}

// LogLookUp Memory allocation
impl<L: AirParameters> AirBuilder<L> {
    pub fn lookup_log_derivative(
        &mut self,
        table: &ElementRegister,
        values: &ArrayRegister<ElementRegister>,
        table_index: fn(L::Field) -> usize,
    ) {
        // Allocate memory for the lookup
        let challenge = self.alloc_challenge::<ExtensionRegister<3>>();
        let multiplicities = self.alloc_array::<ElementRegister>(1);
        let multiplicity_table_log = self.alloc::<ExtensionRegister<3>>();
        let row_accumulators = self.alloc_array::<ExtensionRegister<3>>(values.len() / 2);
        let log_lookup_accumulator = self.alloc::<ExtensionRegister<3>>();
        let table = ArrayRegister::from_element(*table);

        let lookup_data = Lookup::LogDerivative(LogLookup {
            challenge,
            table,
            values: *values,
            multiplicities,
            multiplicity_table_log,
            row_accumulators,
            log_lookup_accumulator,
            table_index,
            _marker: core::marker::PhantomData,
        });

        // Add the lookup constraints
        self.constraints
            .push(Constraint::lookup(lookup_data.clone()));

        // Add the lookup to the list of lookups
        self.lookup_data.push(lookup_data);
    }
}
