use itertools::Itertools;

use super::{LogLookup, LookupTable};
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::cubic::element::CubicElement;
use crate::plonky2::field::cubic::extension::CubicExtension;

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_multiplicities_from_fn<E: CubicParameters<F>, T: Register>(
        &self,
        num_rows: usize,
        table_data: &LookupTable<T, F, E>,
        table_index: impl Fn(T::Value<F>) -> usize,
        values: &[T],
    ) {
        // Calculate multiplicities
        let mut multiplicities = vec![F::ZERO; num_rows];

        let trace = self.read_trace().unwrap();

        for row in trace.rows() {
            for value in values.iter() {
                let val = value.read_from_slice(row);
                let index = table_index(val);
                assert!(index < num_rows);
                multiplicities[index] += F::ONE;
            }
        }
        drop(trace);

        // Write multiplicities into the trace
        let multiplicity = table_data.multiplicities.get(0);
        for (i, mult) in multiplicities.iter().enumerate() {
            self.write(&multiplicity, mult, i);
        }
    }

    /// Writte the table inverses and accumulate
    /// Assumes table multiplicities have been written
    pub(crate) fn write_log_lookup_table<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        table_data: &LookupTable<T, F, E>,
    ) -> Vec<CubicExtension<F, E>> {
        let beta = CubicExtension::<F, E>::from(self.read(&table_data.challenge, 0));
        let mult_table_log_entries = self.write_trace()
            .unwrap()
            .rows_par_mut()
            .map(|row| {
                let mut sum = CubicExtension::ZERO;
                for ((table, multiplicity), table_log_register) in table_data
                    .table
                    .iter()
                    .zip_eq(table_data.multiplicities.iter())
                    .zip_eq(table_data.multiplicities_table_log.iter())
                {
                    let table_val = table.read_from_slice(row);
                    let mult_val = multiplicity.read_from_slice(row);
                    let table = CubicExtension::from(T::trace_value_as_cubic(table_val));
                    let mult = CubicExtension::from(mult_val);
                    let table_log = mult / (beta - table);
                    table_log_register.assign_to_raw_slice(row, &table_log.0);
                    sum += table_log;
                }
                sum
            })
            .collect::<Vec<_>>();

        // Write accumulation
        let mut acc = CubicExtension::ZERO;
        for (i, mult_table) in mult_table_log_entries.iter().enumerate() {
            acc = acc + *mult_table;
            self.write(&table_data.table_accumulator, &acc.0, i);
        }

        mult_table_log_entries
    }

    pub(crate) fn write_log_lookup<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        lookup_data: &LogLookup<T, F, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&lookup_data.challenge, 0));

        // Write multiplicity inverse constraints
        let mult_table_log_entries = self.write_log_lookup_table(&lookup_data.table_data);


        // let split_data = SplitData::split_log_data(lookup_data);
        let accumulators = self
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .map(|row| {
                let mut accumumulator = CubicExtension::ZERO;
                let accumulators = lookup_data.row_accumulators;
                for (k, pair) in lookup_data.values.chunks_exact(2).enumerate() {
                    let a = T::trace_value_as_cubic(pair[0].read_from_slice(row));
                    let b = T::trace_value_as_cubic(pair[1].read_from_slice(row));
                    let beta_minus_a = beta - CubicExtension::from(a);
                    let beta_minus_b = beta - CubicExtension::from(b);
                    accumumulator += beta_minus_a.inverse() + beta_minus_b.inverse();
                    accumulators
                        .get(k)
                        .assign_to_raw_slice(row, &accumumulator.0);
                }
                accumumulator
            })
            .collect::<Vec<_>>();

        let log_lookup_next = lookup_data.log_lookup_accumulator.next();  
        let mut value = CubicExtension::ZERO;
        for (i, (acc, mult_table)) in accumulators
            .into_iter()
            .zip_eq(mult_table_log_entries)
            .enumerate()
            .filter(|(i, _)| *i != num_rows - 1)
        {
            value += acc;
            self.write(&log_lookup_next, &value.0, i);
        }
    }
}
