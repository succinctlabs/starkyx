use itertools::Itertools;

use super::{LogLookup, LogLookupValues, LookupTable};
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
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
        num_rows: usize,
        table_data: &LookupTable<T, F, E>,
    ) -> Vec<CubicExtension<F, E>> {
        let beta = CubicExtension::<F, E>::from(self.read(&table_data.challenge, 0));
        assert_eq!(
            table_data.table.len(),
            table_data.multiplicities_table_log.len()
        );
        assert_eq!(table_data.table.len(), table_data.multiplicities.len());
        let mult_table_log_entries = self
            .write_trace()
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
            acc += *mult_table;
            self.write(&table_data.table_accumulator, &acc.0, i);
        }

        // Write the digest value
        self.write(&table_data.digest, &acc.0, num_rows - 1);

        mult_table_log_entries
    }

    pub(crate) fn write_log_lookup_values<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        values_data: &LogLookupValues<T, F, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&values_data.challenge, 0));

        let accumulators = self
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .map(|row| {
                let mut accumumulator = CubicExtension::ZERO;
                let accumulators = values_data.row_accumulators;
                for (k, pair) in values_data.values.chunks_exact(2).enumerate() {
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

        let log_lookup = values_data.log_lookup_accumulator;
        let mut value = CubicExtension::ZERO;
        for (i, acc) in accumulators.into_iter().enumerate() {
            value += acc;
            self.write(&log_lookup, &value.0, i);
        }

        // Write the digest value
        self.write(&values_data.digest, &value.0, num_rows - 1);
    }

    pub(crate) fn write_log_lookup<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        lookup_data: &LogLookup<T, F, E>,
    ) {
        // let beta = CubicExtension::<F, E>::from(self.read(&lookup_data.challenge, 0));

        // Write multiplicity inverse constraints
        self.write_log_lookup_table(num_rows, &lookup_data.table_data);

        // Write the value data accumulating 1/(beta-value)
        self.write_log_lookup_values(num_rows, &lookup_data.values_data);
    }
}
