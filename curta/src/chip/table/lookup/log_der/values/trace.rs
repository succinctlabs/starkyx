use crate::chip::register::Register;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::math::prelude::cubic::extension::CubicExtension;

use super::LogLookupValues;

use crate::maybe_rayon::*;

impl<F: PrimeField> TraceWriter<F> {

    pub(crate) fn write_log_lookup_values<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        values_data: &LogLookupValues<T, F, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&values_data.challenge, 0));

        // Accumulate lookup values in the trace
        let accumulators = self
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .map(|row| {
                let mut accumumulator = CubicExtension::ZERO;
                let accumulators = values_data.row_accumulators;
                for (k, pair) in values_data.trace_values.chunks_exact(2).enumerate() {
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
        // Write the local digest
        self.write(&values_data.local_digest, &value.0, num_rows - 1);

        // Accumulate lookups for public inputs
        let mut global_accumumulator = CubicExtension::ZERO;
        let global_accumulators = values_data.global_accumulators;
        for (k, pair) in values_data.public_values.chunks_exact(2).enumerate() {
            let a = T::trace_value_as_cubic(self.read(&pair[0], 0));
            let b = T::trace_value_as_cubic(self.read(&pair[1], 0));
            let beta_minus_a = beta - CubicExtension::from(a);
            let beta_minus_b = beta - CubicExtension::from(b);
            global_accumumulator += beta_minus_a.inverse() + beta_minus_b.inverse();
            self.write(&global_accumulators.get(k), &global_accumumulator.0, 0);
        }
        // Write the global digest if exists
        if let Some(global_digest) = values_data.global_digest {
            self.write(&global_digest, &global_accumumulator.0, 0);
        }

        value += global_accumumulator;

        // Write the digest value
        self.write(&values_data.digest, &value.0, num_rows - 1);
    }

}
