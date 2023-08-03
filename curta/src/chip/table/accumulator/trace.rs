use super::Accumulator;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::cubic::element::CubicElement;

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_accumulation<E: CubicParameters<F>>(
        &self,
        accumulator: &Accumulator<F, E>,
    ) {
        let challenges = self.read_vec(&accumulator.challenges, 0);

        // Write accumulation to the digest
        self.write_trace().unwrap().rows_par_mut().for_each(|row| {
            let acc = accumulator
                .values
                .iter()
                .flat_map(|x| x.read_from_slice(row))
                .zip(challenges.iter())
                .map(|(val, alpha)| *alpha * val)
                .sum::<CubicElement<F>>();
            accumulator.digest.assign_to_raw_slice(row, &acc);
        });
    }
}
