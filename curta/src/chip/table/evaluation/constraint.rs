//! Constraints for the evaluation argument
//!
//! evaluation_column.next = evaluation_column+ \sum_{i=0}^{n-1} \alpha_i * value[i]
//!
//!

use super::Evaluation;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::{Register, RegisterSerializable};
use crate::plonky2::field::cubic::element::CubicElement;
use crate::plonky2::field::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for Evaluation<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        // let filter_vec = self.filter.eval(parser);
        // assert_eq!(filter_vec.len(), 1);
        // let filter_base = filter_vec[0];
        // let not_filter_base = parser.sub(parser.one(), filter);
        // let filter = parser.from_base_field(filter_base);
        // let not_filter = parser.from_base_field(not_filter_base);

        // // Constrain the running evaluation powers
        // let beta = self.beta.eval_extension(parser);
        // let beta_powers = self.beta_powers.eval_extension(parser);

        // let powers_minus_beta = parser.sub_extension(beta, beta_powers);
        // parser.constraint_extension_first_row(powers_minus_beta);

        // let beta_powers_next = self.beta_powers.next().eval_extension(parser);
        // let beta_powers_times_beta = parser.mul_extension(beta_powers, beta);
        // let beta_acc_transition_constraint_all =
        //     parser.sub_extension(beta_powers_next, beta_powers_times_beta);
        // let beta_acc_constraint =
        //     parser.mul_extension(beta_acc_transition_constraint_all, filter);

        // let beta_same_all = parser.sub_extension(beta_powers, beta);
        // let beta_same_constraint = parser.mul_extension(beta_same_all, not_filter);
        // parser.constraint_extension_transition(beta_transition_constraint);

        // // Constrain the accumulation
        // let alphas = self
        //     .alphas
        //     .eval_vec(parser)
        //     .into_iter()
        //     .map(|x| CubicElement(x))
        //     .collect::<Vec<_>>();
        // assert_eq!(
        //     alphas.len(),
        //     self.values.len(),
        //     "alphas.len() != self.values.len()"
        // );
        // let mut acc = parser.zero_extension();
        // for (alpha, value) in alphas.iter().zip(self.values.iter()) {
        //     let val = parser.from_base_field(value.eval(parser));
        //     let alpha_times_value = parser.mul_extension(*alpha, val);
        //     acc = parser.add_extension(acc, alpha_times_value);
        // }
        // let accumulator = self.accumulator.eval_extension(parser);
        // let accumulator_next = self.accumulator.next().eval_extension(parser);
        // let acc_diff = parser.sub_extension(accumulator_next, accumulator);
        // let accumulator_constraint_all = parser.sub_extension(acc_diff, acc);
        // let accumulator_constraint = parser.mul_extension(accumulator_constraint_all, filter);
        // parser.constraint_extension_transition(accumulator_constraint);
    }
}
