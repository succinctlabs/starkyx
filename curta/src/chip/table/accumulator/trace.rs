
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

use super::Accumulator;

impl<F: PrimeField> TraceWriter<F> {

    pub(crate) fn write_accumulation<E: CubicParameters<F>>(&self, num_rows: usize, accumulator: &Accumulator<E>, challenges: &[cubic::extension::CubicExtension<F, E>],) {
        todo!()
    }

}