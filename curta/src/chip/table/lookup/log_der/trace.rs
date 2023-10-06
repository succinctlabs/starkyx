use super::{LogLookup, LogLookupTable};
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

impl<F: PrimeField> TraceWriter<F> {
    pub fn write_multiplicities_from_fn<E: CubicParameters<F>, T: Register>(
        &self,
        num_rows: usize,
        table_data: &LogLookupTable<T, F, E>,
        table_index: impl Fn(T::Value<F>) -> usize,
        trace_values: &[T],
        public_values: &[T],
    ) {
        // Calculate multiplicities
        let mut multiplicities = vec![F::ZERO; num_rows];

        // Count the multiplicities in the trace
        let trace = self.read_trace().unwrap();
        for row in trace.rows() {
            for value in trace_values.iter() {
                let val = value.read_from_slice(row);
                let index = table_index(val);
                assert!(index < num_rows);
                multiplicities[index] += F::ONE;
            }
        }
        drop(trace);

        // Count the multiplicities in the public values
        let public_slice = self.public.read().unwrap();
        for value in public_values.iter() {
            let val = value.read_from_slice(&public_slice);
            let index = table_index(val);
            assert!(index < num_rows);
            multiplicities[index] += F::ONE;
        }

        // Write multiplicities into the trace
        let multiplicity = table_data.multiplicities.get(0);
        for (i, mult) in multiplicities.iter().enumerate() {
            self.write(&multiplicity, mult, i);
        }
    }



    pub(crate) fn write_log_lookup<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        lookup_data: &LogLookup<T, F, E>,
    ) {
        // Write multiplicity inverse constraints
        self.write_log_lookup_table(num_rows, &lookup_data.table_data);

        // Write the value data accumulating 1/(beta-value)
        self.write_log_lookup_values(num_rows, &lookup_data.values_data);
    }
}
