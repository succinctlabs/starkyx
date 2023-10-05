use itertools::Itertools;

use super::LogLookupTable;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;

impl<T: EvalCubic, E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LogLookupTable<T, AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let beta = self.challenge.eval(parser);

        let multiplicities = self
            .multiplicities
            .eval_vec(parser)
            .into_iter()
            .map(|e| parser.element_from_base_field(e))
            .collect::<Vec<_>>();

        let table = self
            .table
            .iter()
            .map(|x| x.eval_cubic(parser))
            .collect::<Vec<_>>();

        let multiplicities_table_log = self.multiplicities_table_log.eval_vec(parser);
        let beta_minus_tables = table
            .iter()
            .map(|t| parser.sub_extension(beta, *t))
            .collect::<Vec<_>>();

        // Constrain multiplicities_table_log = sum(mult_i * log(beta - table_i))
        for ((mult_table_log, beta_minus_table), mult) in multiplicities_table_log
            .iter()
            .zip_eq(beta_minus_tables.iter())
            .zip_eq(multiplicities.iter())
        {
            let mult_log_inv_times_table = parser.mul_extension(*mult_table_log, *beta_minus_table);
            let mult_table_constraint = parser.sub_extension(*mult, mult_log_inv_times_table);
            parser.constraint_extension(mult_table_constraint);
        }

        // Constrain the accumulation
        let mult_table_log_sum = multiplicities_table_log
            .iter()
            .fold(parser.zero_extension(), |acc, mult_table_log| {
                parser.add_extension(acc, *mult_table_log)
            });

        let accumulator = self.table_accumulator.eval(parser);

        let first_row_acc = parser.sub_extension(accumulator, mult_table_log_sum);
        parser.constraint_extension_first_row(first_row_acc);

        let mult_table_log_sum_next = self.multiplicities_table_log.iter().fold(
            parser.zero_extension(),
            |acc, mult_table_log| {
                let value = mult_table_log.next().eval(parser);
                parser.add_extension(acc, value)
            },
        );

        let acuumulator_next = self.table_accumulator.next().eval(parser);

        let acc_next_expected = parser.add_extension(accumulator, mult_table_log_sum_next);
        let acc_next_constraint = parser.sub_extension(acuumulator_next, acc_next_expected);
        parser.constraint_extension_transition(acc_next_constraint);

        // Constraint the digest
        let digest = self.digest.eval(parser);
        let digest_constraint = parser.sub_extension(digest, accumulator);
        parser.constraint_extension_last_row(digest_constraint);
    }
}
