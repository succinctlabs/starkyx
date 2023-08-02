use super::{LogLookup, LookupTable};
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::{Register, RegisterSerializable};
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

    pub(crate) fn write_log_lookup_table<T: EvalCubic, E: CubicParameters<F>>(&self, num_rows: usize, table_data: &LookupTable<T, F, E>) {

    }

    pub(crate) fn write_log_lookup<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        lookup_data: &LogLookup<T, F, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&lookup_data.challenge, 0));

        // Write multiplicity inverse constraints
        let mult_table_log_entries = (0..num_rows)
            .into_par_iter()
            .map(|i| {
                let multiplicity_reg = lookup_data.table_data.multiplicities.get(0);
                let x = self.read(&multiplicity_reg, i);
                let table_value = self.read(&lookup_data.table_data.table[0], i);
                let table = CubicExtension::from(T::trace_value_as_cubic(table_value));
                CubicExtension::from(x) / (beta - table)
            })
            .collect::<Vec<_>>();

        let mult_table_log = lookup_data.table_data.multiplicities_table_log.get(0);
        for (i, value) in mult_table_log_entries.iter().enumerate() {
            self.write_slice(&mult_table_log, value.as_base_slice(), i);
        }

        // Log accumulator
        let mut value = CubicExtension::ZERO;
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
        for (i, (acc, mult_table)) in accumulators
            .into_iter()
            .zip(mult_table_log_entries)
            .enumerate()
            .filter(|(i, _)| *i != num_rows - 1)
        {
            value += acc - mult_table;
            self.write_slice(&log_lookup_next, value.as_base_slice(), i);
        }
    }
}
