use alloc::collections::VecDeque;

use itertools::Itertools;

use super::Evaluation;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::cubic::extension::CubicExtension;

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_evaluation<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        evaluation_data: &Evaluation<F, E>,
        beta: CubicExtension<F, E>,
        alphas: Vec<CubicExtension<F, E>>,
    ) {
        let filters = (0..num_rows)
            .into_par_iter()
            .map(|i| self.read_expression(&evaluation_data.filter, i)[0])
            .map(|x| x == F::ZERO)
            .collect::<Vec<_>>();

        let mut acc_values = (0..num_rows)
            .into_par_iter()
            .zip(filters.par_iter())
            .filter(|(_, &filter)| filter)
            .map(|(i, _)| {
                evaluation_data
                    .values
                    .iter()
                    .zip(alphas.iter())
                    .map(|(v, a)| *a * self.read(v, i))
                    .sum::<CubicExtension<F, E>>()
            })
            .collect::<VecDeque<_>>();

        let mut beta_power = CubicExtension::<F, E>::ONE;
        let mut acc = CubicExtension::<F, E>::ZERO;
        for (i, &f) in filters.iter().enumerate() {
            if f {
                let value = acc_values.pop_front().unwrap();
                acc += value * beta_power;
                beta_power *= beta;
            }
            self.write_value(&evaluation_data.accumulator, &acc.base_field_array(), i);
            self.write_value(
                &evaluation_data.beta_powers,
                &beta_power.base_field_array(),
                i,
            );
        }
    }
}
