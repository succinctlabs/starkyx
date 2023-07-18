//! Constraints for the evaluation argument
//!
//! evaluation_column.next = evaluation_column+ \sum_{i=0}^{n-1} \alpha_i * value[i]
//!
//!

use super::Evaluation;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;
use crate::plonky2::field::cubic::element::CubicElement;
use crate::plonky2::field::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for Evaluation<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let filter_vec = self.filter.eval(parser);
        assert_eq!(filter_vec.len(), 1);
        let filter_base = filter_vec[0];
        let not_filter_base = parser.sub_const(filter_base, AP::Field::ONE);
        let filter = parser.from_base_field(filter_base);
        let not_filter = parser.from_base_field(not_filter_base);

        // Constrain the running evaluation powers
        let beta = self.beta.eval_extension(parser);
        let beta_powers = self.beta_powers.eval_extension(parser);

        let one = parser.one_extension();
        let powers_minus_one = parser.sub_extension(beta_powers, one);
        parser.constraint_extension_first_row(powers_minus_one);

        // Constraint
        // (Beta_next - beta * beta_local) * filter
        let beta_powers_next = self.beta_powers.next().eval_extension(parser);
        let beta_powers_times_beta = parser.mul_extension(beta_powers, beta);
        let beta_acc_transition_constraint_all =
            parser.sub_extension(beta_powers_next, beta_powers_times_beta);
        let beta_acc_constraint = parser.mul_extension(beta_acc_transition_constraint_all, filter);

        let beta_same_all = parser.sub_extension(beta_powers_next, beta_powers);
        let beta_same_constraint = parser.mul_extension(beta_same_all, not_filter);

        let beta_constraint = parser.add_extension(beta_acc_constraint, beta_same_constraint);
        parser.constraint_extension_transition(beta_constraint);

        // Constrain the accumulation

        // Constrain first row value
        let accumulator = self.accumulator.eval_extension(parser);
        let zero = parser.zero_extension();
        let acc_first = parser.sub_extension(accumulator, zero);
        parser.constraint_extension_first_row(acc_first);

        // Calculate the accumulated value of the row
        // acc = beta_powers * (\sum_i alpha_i * value_i)
        let alphas = self
            .alphas
            .eval_vec(parser)
            .into_iter()
            .map(|x| CubicElement(x))
            .collect::<Vec<_>>();
        assert_eq!(
            alphas.len(),
            self.values.len(),
            "alphas.len() != self.values.len()"
        );
        let mut row_acc = parser.zero_extension();
        for (alpha, value) in alphas.iter().zip(self.values.iter()) {
            let val = parser.from_base_field(value.eval(parser));
            let alpha_times_value = parser.mul_extension(*alpha, val);
            row_acc = parser.add_extension(row_acc, alpha_times_value);
        }
        let beta_power_row_acc = parser.mul_extension(row_acc, beta_powers);

        // Constrain the transition
        let accumulator_next = self.accumulator.next().eval_extension(parser);
        let expected_acc_next = parser.add_extension(accumulator, beta_power_row_acc);
        let accumulator_constraint_trans_all =
            parser.sub_extension(accumulator_next, expected_acc_next);
        parser.constraint_extension_transition(accumulator_constraint_trans_all);
        let accumulator_constraint_trans =
            parser.mul_extension(accumulator_constraint_trans_all, filter);

        let accumulator_same_all = parser.sub_extension(accumulator_next, accumulator);
        let accumulator_same_constraint = parser.mul_extension(accumulator_same_all, not_filter);

        let accumulator_constraint =
            parser.add_extension(accumulator_constraint_trans, accumulator_same_constraint);

        parser.constraint_extension_transition(accumulator_constraint);

        // last row constraint, digest
        let digest = self.digest.eval_extension(parser);
        let digest_const_all = parser.sub_extension(digest, expected_acc_next);
        let digest_acc_constraint = parser.mul_extension(digest_const_all, filter);

        let digest_same = parser.sub_extension(digest, accumulator);
        let digest_same_constraint = parser.mul_extension(digest_same, not_filter);

        let digest_constraint = parser.add_extension(digest_acc_constraint, digest_same_constraint);
        parser.constraint_extension_last_row(digest_constraint);
    }
}
