//! Constraints for the evaluation argument
//!
//! evaluation_column.next = evaluation_column+ \sum_{i=0}^{n-1} \alpha_i * value[i]
//!
//!


use super::{Digest, Evaluation};
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;
use crate::plonky2::field::cubic::element::CubicElement;
use crate::plonky2::field::CubicParameters;

impl<F: Field, E: CubicParameters<F>> Digest<F, E> {
    fn eval_digest<AP: CubicParser<E, Field = F>>(
        &self,
        parser: &mut AP,
        digest_val: CubicElement<<AP as AirParser>::Var>,
        alphas: &[CubicElement<<AP as AirParser>::Var>],
        beta: CubicElement<<AP as AirParser>::Var>,
    ) {
        match self {
            Digest::None => {}
            Digest::Extended(register) => {
                let digest = register.eval_extension(parser);
                let constraint = parser.sub_extension(digest, digest_val);
                parser.constraint_extension_last_row(constraint);
            }
            Digest::Values(values) => {
                let mut digest = parser.zero_extension();
                let mut beta_powers = parser.one_extension();
                for array in values {
                    assert_eq!(array.len(), alphas.len());
                    // calculate value
                    let mut acc = parser.zero_extension();
                    for (reg, alpha) in array.into_iter().zip(alphas.iter()) {
                        let reg_val = reg.eval(parser);
                        let reg_val_extension = parser.from_base_field(reg_val);
                        let alpha_times_val = parser.mul_extension(*alpha, reg_val_extension);
                        acc = parser.add_extension(acc, alpha_times_val);
                    }
                    let acc_times_beta = parser.mul_extension(acc, beta_powers);
                    digest = parser.add_extension(digest, acc_times_beta);
                    beta_powers = parser.mul_extension(beta_powers, beta);
                }
                let digest_constraint = parser.sub_extension(digest, digest_val);
                parser.constraint_extension_last_row(digest_constraint);
            }
            _ => unimplemented!(),
        }
    }
}

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for Evaluation<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let filter_vec = self.filter.eval(parser);
        assert_eq!(filter_vec.len(), 1);
        let filter_base = filter_vec[0];
        let one = parser.one();
        let not_filter_base = parser.sub(one, filter_base);
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
        let beta_powers_next_filter = parser.mul_extension(beta_powers_times_beta, filter);

        let beta_powers_next_not_filter = parser.mul_extension(beta_powers, not_filter);
        let beta_powers_next_value =
            parser.add_extension(beta_powers_next_filter, beta_powers_next_not_filter);

        let beta_powers_next_constraint =
            parser.sub_extension(beta_powers_next, beta_powers_next_value);
        parser.constraint_extension_transition(beta_powers_next_constraint);

        // Constrain the accumulation

        // Constrain first row value
        let accumulator = self.accumulator.eval_extension(parser);
        parser.constraint_extension_first_row(accumulator);

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
        // constrain row accumulator
        let row_beta_powers = parser.mul_extension(beta_powers, row_acc);
        let row_accumulator = self.row_accumulator.eval_extension(parser);
        let row_acc_constraint = parser.sub_extension(row_accumulator, row_beta_powers);
        parser.constraint_extension(row_acc_constraint);

        // Constrain the transition
        let accumulator_next = self.accumulator.next().eval_extension(parser);
        let acc_next_filter_val = parser.add_extension(accumulator, row_accumulator);
        let acc_next_filter = parser.mul_extension(acc_next_filter_val, filter);

        let acc_next_not_filter = parser.mul_extension(accumulator, not_filter);
        let accumulator_next_value = parser.add_extension(acc_next_filter, acc_next_not_filter);
        let accumulator_constraint = parser.sub_extension(accumulator_next, accumulator_next_value);
        parser.constraint_extension_transition(accumulator_constraint);

        // last row constraint, digest
        self.digest.eval_digest(parser, accumulator_next_value, &alphas, beta);
    }
}
