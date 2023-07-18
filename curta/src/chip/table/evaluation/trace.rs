use alloc::collections::VecDeque;

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
        alphas: &[CubicExtension<F, E>],
    ) {
        let filters = (0..num_rows)
            .into_par_iter()
            .map(|i| self.read_expression(&evaluation_data.filter, i)[0])
            .map(|x| x == F::ONE)
            .collect::<Vec<_>>();

        let acc_values = (0..num_rows)
            .into_par_iter()
            .map(|i| {
                let value = evaluation_data
                    .values
                    .iter()
                    .zip(alphas.iter())
                    .map(|(v, a)| *a * self.read(v, i))
                    .sum::<CubicExtension<F, E>>();
                self.write(
                    &evaluation_data.row_accumulator,
                    &value.base_field_array(),
                    i,
                );
                value
            })
            .collect::<Vec<_>>();

        let mut beta_power = CubicExtension::<F, E>::ONE;
        let mut acc = CubicExtension::<F, E>::ZERO;
        for (i, &f) in filters.iter().enumerate() {
            self.write_value(
                &evaluation_data.beta_powers,
                &beta_power.base_field_array(),
                i,
            );

            self.write_value(&evaluation_data.accumulator, &acc.base_field_array(), i);

            if f {
                let value = acc_values[i];
                acc += value * beta_power;
                beta_power *= beta;
            }
        }
        debug_assert_eq!(acc_values.len(), 0);
        self.write_value(
            &evaluation_data.digest,
            &acc.base_field_array(),
            num_rows - 1,
        );
    }
}
