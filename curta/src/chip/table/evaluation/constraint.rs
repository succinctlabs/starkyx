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
use crate::plonky2::field::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for Evaluation<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let filter_vec = self.filter.eval(parser);
        assert_eq!(filter_vec.len(), 1);
        let filter = filter_vec[0];

        // Constrain the running evaluation powers
        let beta = self.beta.eval_extension(parser);
        let beta_powers = self.beta_powers.eval_extension(parser);

        let powers_minus_beta = parser.sub_extension(beta, beta_powers);
        parser.constraint_extension_first_row(powers_minus_beta);

        let beta_powers_next = self.beta_powers.next().eval_extension(parser);
        let beta_powers_times_beta = parser.mul_extension(beta_powers, beta);
        let beta_transition_constraint =
            parser.sub_extension(beta_powers_next, beta_powers_times_beta);
        parser.constraint_extension_transition(beta_transition_constraint);

        // Constrain the accumulation
    }
}
