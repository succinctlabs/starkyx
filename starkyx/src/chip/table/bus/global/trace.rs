use super::Bus;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::trace::writer::TraceWriter;
use crate::math::extension::cubic::extension::CubicExtension;
use crate::math::prelude::*;

impl<F: PrimeField> TraceWriter<F> {
    pub fn write_global_bus<T: EvalCubic, E: CubicParameters<F>>(&self, bus: &Bus<T, E>) {
        let beta = CubicExtension::<F, E>::from(self.read(&bus.challenge, 0));

        // Accumulate bus entries for global values
        self.write_log_global_accumulation(
            beta,
            &bus.global_entries,
            &bus.global_accumulators,
            bus.global_value,
        );
    }
}
