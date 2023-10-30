use super::BusChannel;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::trace::writer::TraceWriter;
use crate::math::extension::cubic::extension::CubicExtension;
use crate::math::prelude::*;

impl<F: PrimeField> TraceWriter<F> {
    pub fn write_bus_channel<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        channel: &BusChannel<T, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&channel.challenge, 0));

        // Accumulate bus values in the trace.
        let accumulated_bus_value = self.write_log_trace_accumulation(
            beta,
            &channel.entries,
            &channel.row_accumulators,
            channel.table_accumulator,
        );

        // Write the final accumulated value to the output channel
        self.write(
            &channel.out_channel,
            &accumulated_bus_value.0,
            self.height - 1,
        );
    }
}
