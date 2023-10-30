use super::entry::LogEntry;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::slice::RegisterSlice;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::cubic::extension::CubicExtension;
use crate::math::prelude::*;
use crate::maybe_rayon::*;

impl<F: PrimeField> TraceWriter<F> {
    pub fn write_log_trace_accumulation<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        beta: CubicExtension<F, E>,
        entries: &[LogEntry<T>],
        intermediate_values: &impl RegisterSlice<CubicRegister>,
        trace_accumulator: CubicRegister,
    ) -> CubicExtension<F, E> {
        // Accumulate lookup values in the trace
        let accumulators = self
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .map(|row| {
                let entry_chunks = entries.chunks_exact(2);
                let last_element = entry_chunks
                    .remainder()
                    .first()
                    .map(|reg| reg.read_from_slice(row).evaluate(beta))
                    .unwrap_or(CubicExtension::ZERO);
                let mut accumumulator = CubicExtension::ZERO;
                let accumulators = intermediate_values;
                for (k, pair) in entry_chunks.enumerate() {
                    let a = pair[0].read_from_slice(row);
                    let b = pair[1].read_from_slice(row);
                    accumumulator += a.evaluate(beta) + b.evaluate(beta);
                    accumulators
                        .get_value(k)
                        .assign_to_raw_slice(row, &accumumulator.0);
                }
                accumumulator + last_element
            })
            .collect::<Vec<_>>();

        let mut value = CubicExtension::ZERO;
        for (i, acc) in accumulators.into_iter().enumerate() {
            value += acc;
            self.write(&trace_accumulator, &value.0, i);
        }
        // Write the local digest
        self.write(&trace_accumulator, &value.0, self.height - 1);

        value
    }

    pub fn write_log_global_accumulation<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        beta: CubicExtension<F, E>,
        entries: &[LogEntry<T>],
        intermediate_values: &impl RegisterSlice<CubicRegister>,
        global_accumulator: CubicRegister,
    ) -> CubicExtension<F, E> {
        let value_chunks = entries.chunks_exact(2);
        let last_element = value_chunks
            .remainder()
            .last()
            .map(|reg| self.read_log_entry(reg, 0).evaluate(beta))
            .unwrap_or(CubicExtension::ZERO);
        let mut accumumulator = CubicExtension::ZERO;
        for (k, pair) in value_chunks.enumerate() {
            let a = self.read_log_entry(&pair[0], 0);
            let b = self.read_log_entry(&pair[1], 0);
            accumumulator += a.evaluate(beta) + b.evaluate(beta);
            self.write(&intermediate_values.get_value(k), &accumumulator.0, 0);
        }
        let value = accumumulator + last_element;
        self.write(&global_accumulator, &value.0, 0);

        value
    }
}
