use super::LogLookup;
use crate::chip::register::RegisterSerializable;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::cubic::extension::CubicExtension;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitData {
    mid: usize,
    values_range: (usize, usize),
    acc_range: (usize, usize),
}

impl SplitData {
    pub fn new(mid: usize, values_range: (usize, usize), acc_range: (usize, usize)) -> Self {
        Self {
            mid,
            values_range,
            acc_range,
        }
    }
    pub(crate) fn split<'a, T>(&self, trace_row: &'a mut [T]) -> (&'a [T], &'a mut [T]) {
        let (left, right) = trace_row.split_at_mut(self.mid);
        (
            &left[self.values_range.0..self.values_range.1],
            &mut right[self.acc_range.0..self.acc_range.1],
        )
    }

    pub(crate) fn split_data<F: Field, E: CubicParameters<F>>(
        log_data: &LogLookup<F, E, 1>,
    ) -> Self {
        let values_idx = log_data.values.register().get_range();
        let acc_idx = log_data.row_accumulators.register().get_range();
        assert!(
            values_idx.0 < acc_idx.0,
            "Illegal memory pattern, expected values indices \
        to be to the right of accumulator indices, \
        instead got: values_idx: {:?}, acc_idx: {:?}",
            values_idx,
            acc_idx
        );
        SplitData::new(
            values_idx.1,
            (values_idx.0, values_idx.1),
            (acc_idx.0 - values_idx.1, acc_idx.1 - values_idx.1),
        )
    }
}

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_log_lookup<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        lookup_data: &LogLookup<F, E, 1>,
        beta: CubicExtension<F, E>,
    ) {
        let table_index = lookup_data.table_index;
        let values_idx = lookup_data.values.register().get_range();

        // Calculate multiplicities
        let mut multiplicities = vec![F::ZERO; num_rows];

        let trace = self.read_trace().unwrap().clone();

        for row in trace.rows() {
            for value in row[values_idx.0..values_idx.1].iter() {
                let index = table_index(*value);
                assert!(index < num_rows);
                multiplicities[index] += F::ONE;
            }
        }

        // Write multiplicities into the trace
        let multiplicity = lookup_data.multiplicities.get(0);
        for (i, mult) in multiplicities.iter().enumerate() {
            self.write(&multiplicity, &[*mult], i);
        }

        // Write multiplicity inverse constraint
        let mult_table_log_entries = multiplicities
            .par_iter()
            .enumerate()
            .map(|(i, x)| {
                let table = CubicExtension::from(F::from_canonical_usize(i));
                CubicExtension::from(*x) / (beta - table)
            })
            .collect::<Vec<_>>();

        let mult_table_log = lookup_data.multiplicity_table_log;
        for (i, value) in mult_table_log_entries.iter().enumerate() {
            self.write(&mult_table_log, value.as_base_slice(), i);
        }

        // Log accumulator
        let mut value = CubicExtension::ZERO;
        let split_data = SplitData::split_data(lookup_data);
        let accumulators = self
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .map(|row| {
                let (values, accumulators) = split_data.split(row);
                let mut accumumulator = CubicExtension::ZERO;
                for (k, pair) in values.chunks(2).enumerate() {
                    let beta_minus_a = beta - CubicExtension::from(pair[0]);
                    let beta_minus_b = beta - CubicExtension::from(pair[1]);
                    accumumulator += beta_minus_a.inverse() + beta_minus_b.inverse();
                    accumulators[3 * k..3 * k + 3].copy_from_slice(accumumulator.as_base_slice());
                }
                accumumulator
            })
            .collect::<Vec<_>>();

        let log_lookup_next = lookup_data.log_lookup_accumulator.next();
        for (i, (acc, mult_table)) in accumulators
            .into_iter()
            .zip(mult_table_log_entries.into_iter())
            .enumerate()
            .filter(|(i, _)| *i != num_rows - 1)
        {
            value += acc - mult_table;
            self.write(&log_lookup_next, value.as_base_slice(), i);
        }
    }
}
