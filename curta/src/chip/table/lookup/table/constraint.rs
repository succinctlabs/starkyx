use itertools::Itertools;

use super::{LogLookupTable, LookupTable};
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::log_derivative::constraints::LogConstraints;
use crate::chip::table::log_derivative::entry::LogEntry;
use crate::math::prelude::*;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LookupTable<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            LookupTable::Element(t) => t.eval(parser),
            LookupTable::Cubic(t) => t.eval(parser),
        }
    }
}

impl<T: EvalCubic, E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LogLookupTable<T, AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let beta = self.challenge.eval(parser);

        // Constrain multiplicities_table_log = sum(mult_i * log(beta - table_i))
        for ((mult_table_log, table), mult) in self
            .multiplicities_table_log
            .iter()
            .zip_eq(self.table.iter())
            .zip_eq(self.multiplicities)
        {
            let mult_table_log = mult_table_log.eval(parser);
            let table = LogEntry::input_with_multiplicity(*table, mult).eval(parser);
            let mult_table_constraint = LogConstraints::log(parser, beta, table, mult_table_log);
            parser.constraint_extension(mult_table_constraint);
        }

        // Constrain the accumulation
        let mult_table_log_sum = self.multiplicities_table_log.iter().fold(
            parser.zero_extension(),
            |acc, mult_table_log| {
                let mult_table_log = mult_table_log.eval(parser);
                parser.add_extension(acc, mult_table_log)
            },
        );

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
