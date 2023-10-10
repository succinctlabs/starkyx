use super::{LogLookupValues, LookupValues};
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::cubic::extension::CubicExtension;
use crate::math::prelude::*;
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
                let value_chunks = values_data.trace_values.chunks_exact(2);
                let last_element = value_chunks
                    .remainder()
                    .first()
                    .map(|reg| {
                        let r = T::trace_value_as_cubic(reg.value().read_from_slice(row));
                        let beta_minus_a = beta - CubicExtension::from(r);
                        beta_minus_a.inverse()
                    })
                    .unwrap_or(CubicExtension::ZERO);
                let mut accumumulator = CubicExtension::ZERO;
                let accumulators = values_data.row_accumulators;
                for (k, pair) in values_data.trace_values.chunks_exact(2).enumerate() {
                    let a = T::trace_value_as_cubic(pair[0].value().read_from_slice(row));
                    let b = T::trace_value_as_cubic(pair[1].value().read_from_slice(row));
                    let beta_minus_a = beta - CubicExtension::from(a);
                    let beta_minus_b = beta - CubicExtension::from(b);
                    accumumulator += beta_minus_a.inverse() + beta_minus_b.inverse();
                    accumulators
                        .get(k)
                        .assign_to_raw_slice(row, &accumumulator.0);
                }
                accumumulator + last_element
            })
            .collect::<Vec<_>>();

        let log_lookup = values_data.local_digest;
        let mut value = CubicExtension::ZERO;
        for (i, acc) in accumulators.into_iter().enumerate() {
            value += acc;
            self.write(&log_lookup, &value.0, i);
        }
        // Write the local digest
        self.write(&values_data.local_digest, &value.0, num_rows - 1);

        // Accumulate lookups for public inputs
        let global_value_chunks = values_data.public_values.chunks_exact(2);
        let global_last_element = global_value_chunks
            .remainder()
            .last()
            .map(|reg| {
                let r = T::trace_value_as_cubic(self.read(reg.value(), 0));
                let beta_minus_a = beta - CubicExtension::from(r);
                beta_minus_a.inverse()
            })
            .unwrap_or(CubicExtension::ZERO);
        let mut global_accumumulator = CubicExtension::ZERO;
        let global_accumulators = values_data.global_accumulators;
        for (k, pair) in values_data.public_values.chunks_exact(2).enumerate() {
            let a = T::trace_value_as_cubic(self.read(pair[0].value(), 0));
            let b = T::trace_value_as_cubic(self.read(pair[1].value(), 0));
            let beta_minus_a = beta - CubicExtension::from(a);
            let beta_minus_b = beta - CubicExtension::from(b);
            global_accumumulator += beta_minus_a.inverse() + beta_minus_b.inverse();
            self.write(&global_accumulators.get(k), &global_accumumulator.0, 0);
        }
        let global_acc_value = global_accumumulator + global_last_element;
        // Write the global digest if exists
        if let Some(global_digest) = values_data.global_digest {
            self.write(&global_digest, &global_acc_value.0, 0);
        }

        value += global_acc_value;

        // Write the digest value
        self.write(&values_data.digest, &value.0, num_rows - 1);
    }

    pub(crate) fn write_lookup_values<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        values_data: &LookupValues<F, E>,
    ) {
        match values_data {
            LookupValues::Element(values) => {
                self.write_log_lookup_values(num_rows, values);
            }
            LookupValues::Cubic(values) => {
                self.write_log_lookup_values(num_rows, values);
            }
        }
    }
}
