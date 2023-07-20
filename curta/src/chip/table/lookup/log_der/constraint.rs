use alloc::collections::VecDeque;
use core::array;

use super::LogLookup;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>, const N: usize> AirConstraint<AP>
    for LogLookup<AP::Field, E, N>
{
    fn eval(&self, parser: &mut AP) {
        let beta = self.challenge.eval_extension(parser);

        let multiplicities: [_; N] = self
            .multiplicities
            .eval_array(parser)
            .map(|e| parser.element_from_base_field(e));

        let table: [_; N] = self
            .table
            .eval_array(parser)
            .map(|e| parser.element_from_base_field(e));

        let multiplicities_table_log = self.multiplicity_table_log.eval_extension(parser);
        let beta_minus_table: [_; N] = array::from_fn(|i| parser.sub_extension(beta, table[i]));

        // Constrain multiplicities_table_log = sum(mult_i * log(beta - table_i))
        let mult_table_constraint = match N {
            1 => {
                let mult_times_table =
                    parser.mul_extension(multiplicities_table_log, beta_minus_table[0]);
                parser.sub_extension(multiplicities[0], mult_times_table)
            }
            2 => {
                let tables_prod = parser.mul_extension(beta_minus_table[0], beta_minus_table[1]);
                let mult_times_tables = parser.mul_extension(multiplicities_table_log, tables_prod);
                let mult_tablle_0 = parser.mul_extension(multiplicities[0], beta_minus_table[1]);
                let mult_tablle_1 = parser.mul_extension(multiplicities[1], beta_minus_table[0]);
                let numerator = parser.add_extension(mult_tablle_0, mult_tablle_1);
                parser.sub_extension(numerator, mult_times_tables)
            }
            0 => unreachable!("N must be greater than 0"),
            _ => unimplemented!("N > 2 not supported"),
        };
        parser.constraint_extension(mult_table_constraint);

        // Constraint the accumulators for the elements being looked up
        // The accumulators collect the sums of the logarithmic derivatives 1/(beta - element_i)
        let mut row_acc_queue = self
            .row_accumulators
            .iter()
            .map(|x| x.eval_extension(parser))
            .collect::<VecDeque<_>>();

        let mut range_pairs = (0..self.values.len())
            .step_by(2)
            .map(|k| {
                let a_base = self.values.get(k).eval(parser);
                let b_base = self.values.get(k + 1).eval(parser);
                let a = parser.element_from_base_field(a_base);
                let b = parser.element_from_base_field(b_base);
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

        let log_lookup_accumulator = self.log_lookup_accumulator.eval_extension(parser);
        let log_lookup_accumulator_next = self.log_lookup_accumulator.next().eval_extension(parser);

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
