pub mod constraint;
pub mod trace;

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use super::constraint::LookupConstraint;
use super::values::{LogLookupValues, LookupValues};
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::Constraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::Register;
use crate::chip::table::log_derivative::entry::LogEntry;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum LookupTable<F, E> {
    Element(LogLookupTable<ElementRegister, F, E>),
    Cubic(LogLookupTable<CubicRegister, F, E>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogLookupTable<T: Register, F, E> {
    pub(crate) challenge: CubicRegister,
    pub(crate) table: Vec<T>,
    pub(crate) multiplicities: ArrayRegister<ElementRegister>,
    pub(crate) multiplicities_table_log: ArrayRegister<CubicRegister>,
    pub(crate) table_accumulator: CubicRegister,
    pub(crate) digest: CubicRegister,
    pub(crate) values_digests: Vec<CubicRegister>,
    pub(crate) _marker: core::marker::PhantomData<(F, E)>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_lookup<T: EvalCubic>(
        &mut self,
        table: &[T],
        multiplicities: &ArrayRegister<ElementRegister>,
    ) -> LogLookupTable<T, L::Field, L::CubicParams> {
        let challenge = self.alloc_challenge();
        let multiplicities_table_log = self.alloc_array_extended::<CubicRegister>(table.len());
        let table_accumulator = self.alloc_extended();
        let digest = self.alloc_global();

        LogLookupTable {
            challenge,
            table: table.to_vec(),
            multiplicities: *multiplicities,
            multiplicities_table_log,
            table_accumulator,
            digest,
            values_digests: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn constrain_element_lookup_table(
        &mut self,
        table: LogLookupTable<ElementRegister, L::Field, L::CubicParams>,
    ) {
        // insert the table to the builder
        self.lookup_tables.push(LookupTable::Element(table.clone()));

        // Register digest constraints between the table and the lookup values.
        self.global_constraints.push(Constraint::lookup(
            LookupConstraint::<ElementRegister, _, _>::Digest(
                table.digest,
                table.values_digests.clone(),
            )
            .into(),
        ));

        // Register the table constraints.
        self.constraints
            .push(Constraint::lookup(LookupConstraint::Table(table).into()));
    }

    pub fn constrain_cubic_lookup_table(
        &mut self,
        table: LogLookupTable<CubicRegister, L::Field, L::CubicParams>,
    ) {
        // insert the table to the builder
        self.lookup_tables.push(LookupTable::Cubic(table.clone()));

        // Register digest constraints between the table and the lookup values.
        self.global_constraints.push(Constraint::lookup(
            LookupConstraint::<CubicRegister, _, _>::Digest(
                table.digest,
                table.values_digests.clone(),
            )
            .into(),
        ));

        // Register the table constraints.
        self.constraints
            .push(Constraint::lookup(LookupConstraint::Table(table).into()));
    }
}

impl<T: EvalCubic, F: Field, E: CubicParameters<F>> LogLookupTable<T, F, E> {
    pub(crate) fn new_lookup_values<L: AirParameters<Field = F, CubicParams = E>>(
        &mut self,
        builder: &mut AirBuilder<L>,
        values: &[T],
    ) -> LogLookupValues<T, F, E> {
        let mut trace_values = Vec::new();
        let mut public_values = Vec::new();

        for value in values.iter() {
            match value.register() {
                MemorySlice::Public(..) => public_values.push(LogEntry::input(*value)),
                MemorySlice::Local(..) => trace_values.push(LogEntry::input(*value)),
                MemorySlice::Next(..) => unreachable!("Next register not supported for lookup"),
                MemorySlice::Global(..) => public_values.push(LogEntry::input(*value)),
                MemorySlice::Challenge(..) => unreachable!("Cannot lookup challenge register"),
            }
        }

        let row_accumulators =
            builder.alloc_array_extended::<CubicRegister>(trace_values.len() / 2);
        let global_accumulators =
            builder.alloc_array_global::<CubicRegister>(public_values.len() / 2);
        let local_digest = builder.alloc_extended::<CubicRegister>();

        let digest = builder.alloc_global::<CubicRegister>();
        let global_digest = Some(builder.alloc_global::<CubicRegister>());

        self.values_digests.push(digest);

        LogLookupValues {
            challenge: self.challenge,
            trace_values,
            public_values,
            row_accumulators,
            global_accumulators,
            local_digest,
            global_digest,
            digest,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, E: CubicParameters<F>> LogLookupTable<ElementRegister, F, E> {
    pub fn register_lookup_values<L: AirParameters<Field = F, CubicParams = E>>(
        &mut self,
        builder: &mut AirBuilder<L>,
        values: &[ElementRegister],
    ) -> LogLookupValues<ElementRegister, F, E> {
        let lookup_values = self.new_lookup_values(builder, values);
        lookup_values.register_constraints(builder);
        builder
            .lookup_values
            .push(LookupValues::Element(lookup_values.clone()));
        lookup_values
    }
}

impl<F: Field, E: CubicParameters<F>> LogLookupTable<CubicRegister, F, E> {
    pub fn register_lookup_values<L: AirParameters<Field = F, CubicParams = E>>(
        &mut self,
        builder: &mut AirBuilder<L>,
        values: &[CubicRegister],
    ) -> LogLookupValues<CubicRegister, F, E> {
        let lookup_values = self.new_lookup_values(builder, values);
        lookup_values.register_constraints(builder);
        builder
            .lookup_values
            .push(LookupValues::Cubic(lookup_values.clone()));
        lookup_values
    }
}
