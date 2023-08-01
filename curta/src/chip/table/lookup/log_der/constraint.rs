use alloc::collections::VecDeque;

use super::LogLookup;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;

impl<T: EvalCubic, E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LogLookup<T, AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let beta = self.challenge.eval(parser);

        let multiplicities = self
            .table_data
            .multiplicities
            .eval_vec(parser)
            .into_iter()
            .map(|e| parser.element_from_base_field(e))
            .collect::<Vec<_>>();

        let table = self
            .table_data
            .table
            .iter()
            .map(|x| x.eval_cubic(parser))
            .collect::<Vec<_>>();

        let multiplicities_table_log = self.multiplicity_table_log.eval(parser);
        let beta_minus_table = parser.sub_extension(beta, table[0]);

        // Constrain multiplicities_table_log = sum(mult_i * log(beta - table_i))
        let mult_table_constraint = {
            let mult_times_table = parser.mul_extension(multiplicities_table_log, beta_minus_table);
            parser.sub_extension(multiplicities[0], mult_times_table)
        };

        parser.constraint_extension(mult_table_constraint);

        // Constraint the accumulators for the elements being looked up
        // The accumulators collect the sums of the logarithmic derivatives 1/(beta - element_i)
        let mut row_acc_queue = self
            .row_accumulators
            .iter()
            .map(|x| x.eval(parser))
            .collect::<VecDeque<_>>();

        let mut range_pairs = self
            .values
            .chunks_exact(2)
            .map(|chunk| {
                let a = chunk[0].eval_cubic(parser);
                let b = chunk[1].eval_cubic(parser);
                (a, b)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(a, b)| (parser.sub_extension(beta, a), parser.sub_extension(beta, b)))
            .collect::<VecDeque<_>>();

        let ((beta_minus_a_0, beta_minus_b_0), acc_0) = (
            range_pairs.pop_front().unwrap(),
            row_acc_queue.pop_front().unwrap(),
        );

        let beta_minus_a_b = parser.mul_extension(beta_minus_a_0, beta_minus_b_0);
        let acc_beta_m_ab = parser.mul_extension(acc_0, beta_minus_a_b);
        let mut constr_0 = parser.add_extension(beta_minus_a_0, beta_minus_b_0);
        constr_0 = parser.sub_extension(constr_0, acc_beta_m_ab);
        parser.constraint_extension(constr_0);

        let mut prev = acc_0;
        for ((beta_minus_a, beta_minus_b), acc) in range_pairs.iter().zip(row_acc_queue.iter()) {
            let acc_minus_prev = parser.sub_extension(*acc, prev);
            let mut product = parser.mul_extension(*beta_minus_a, *beta_minus_b);
            product = parser.mul_extension(product, acc_minus_prev);
            let mut constraint = parser.add_extension(*beta_minus_a, *beta_minus_b);
            constraint = parser.sub_extension(constraint, product);
            parser.constraint_extension(constraint);
            prev = *acc;
        }

        let log_lookup_accumulator = self.log_lookup_accumulator.eval(parser);
        let log_lookup_accumulator_next = self.log_lookup_accumulator.next().eval(parser);

        let mut acc_transition_constraint =
            parser.sub_extension(log_lookup_accumulator_next, log_lookup_accumulator);
        acc_transition_constraint = parser.sub_extension(acc_transition_constraint, prev);
        acc_transition_constraint =
            parser.add_extension(acc_transition_constraint, multiplicities_table_log);
        parser.constraint_extension(acc_transition_constraint);

        let acc_first_row_constraint = log_lookup_accumulator;
        parser.constraint_extension_first_row(acc_first_row_constraint);

        let mut acc_last_row_constraint = parser.add_extension(log_lookup_accumulator, prev);
        acc_last_row_constraint =
            parser.sub_extension(acc_last_row_constraint, multiplicities_table_log);
        parser.constraint_extension_last_row(acc_last_row_constraint);
    }
}
