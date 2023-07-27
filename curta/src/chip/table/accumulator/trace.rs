use super::Accumulator;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::RegisterSerializable;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::cubic::element::CubicElement;

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_accumulation<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        accumulator: &Accumulator<E>,
    ) {
        let challenges = self.read_vec(&accumulator.challenges, 0);

        // Get an iterator over the trace cells of the values
        let values = accumulator
            .values
            .iter()
            .flat_map(|v| ArrayRegister::<ElementRegister>::from_register_unsafe(*v));

        // Write accumulation to the digest
        (0..num_rows).into_par_iter().for_each(|i| {
            let acc = values
                .clone()
                .map(|x| self.read(&x, i))
                .zip(challenges.iter())
                .map(|(val, alpha)| *alpha * val)
                .sum::<CubicElement<F>>();
            self.write(&accumulator.digest, &acc, i);
        });
    }
}
