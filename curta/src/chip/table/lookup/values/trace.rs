use super::{LogLookupValues, LookupValues};
use crate::chip::register::cubic::EvalCubic;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::cubic::extension::CubicExtension;
use crate::math::prelude::*;

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_log_lookup_values<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        values_data: &LogLookupValues<T, F, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&values_data.challenge, 0));

        // Accumulate lookup values in the trace
        let trace_accumulated_value = self.write_log_trace_accumulation(
            beta,
            &values_data.trace_values,
            &values_data.row_accumulators,
            values_data.local_digest,
        );

        // Accumulate lookups for public inputs
        let global_accumulated_value = if let Some(global_digest) = values_data.global_digest {
            self.write_log_global_accumulation(
                beta,
                &values_data.public_values,
                &values_data.global_accumulators,
                global_digest,
            )
        } else {
            CubicExtension::ZERO
        };

        let value = trace_accumulated_value + global_accumulated_value;

        // Write the total digest value
        self.write(&values_data.digest, &value.0, self.height - 1);
    }

    pub(crate) fn write_lookup_values<E: CubicParameters<F>>(
        &self,
        values_data: &LookupValues<F, E>,
    ) {
        match values_data {
            LookupValues::Element(values) => {
                self.write_log_lookup_values(values);
            }
            LookupValues::Cubic(values) => {
                self.write_log_lookup_values(values);
            }
        }
    }
}
