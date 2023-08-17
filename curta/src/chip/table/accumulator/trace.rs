use super::Accumulator;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::math::extension::cubic::element::CubicElement;
use crate::math::prelude::*;
use crate::maybe_rayon::*;

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_accumulation<E: CubicParameters<F>>(
        &self,
        accumulator: &Accumulator<F, E>,
    ) {
        let challenges = self.read_vec(&accumulator.challenges, 0);

        // Write accumulation to the digest
        match accumulator.digest.register() {
            MemorySlice::Global(..) => {
                let acc = accumulator
                    .values
                    .iter()
                    .flat_map(|x| self.read_expression(x, 0))
                    .zip(challenges.iter())
                    .map(|(val, alpha)| *alpha * val)
                    .sum::<CubicElement<F>>();
                self.write(&accumulator.digest, &acc, 0);
            }
            _ => {
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
    }
}
