use itertools::Itertools;

use super::{LogLookupValues, LookupTable};
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum LookupConstraint<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    Table(LookupTable<T, F, E>),
    ValuesLocal(LogLookupValues<T, F, E>),
    ValuesGlobal(LogLookupValues<T, F, E>),
    ValuesDigest(CubicRegister, CubicRegister, Option<CubicRegister>),
    Digest(CubicRegister, CubicRegister),
}

impl<T: EvalCubic, E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LookupConstraint<T, AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            LookupConstraint::Table(table) => table.eval(parser),
            LookupConstraint::ValuesLocal(values) => values.eval(parser),
            LookupConstraint::ValuesGlobal(values) => values.eval_global(parser),
            LookupConstraint::ValuesDigest(digest, local_digest, global_digest) => {
                let digest = digest.eval(parser);
                let local_digest = local_digest.eval(parser);
                let global_digest = global_digest
                    .map(|d| d.eval(parser))
                    .unwrap_or_else(|| parser.zero_extension());

                let mut digest_constraint = parser.add_extension(local_digest, global_digest);
                digest_constraint = parser.sub_extension(digest_constraint, digest);
                parser.constraint_extension_last_row(digest_constraint);
            }
            LookupConstraint::Digest(a, b) => {
                let a = a.eval_cubic(parser);
                let b = b.eval_cubic(parser);
                let difference = parser.sub_extension(a, b);
                parser.constraint_extension_last_row(difference);
            }
        }
    }
}

impl<T: EvalCubic, E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LookupTable<T, AP::Field, E>
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

impl<T: EvalCubic, F: Field, E: CubicParameters<F>> LogLookupValues<T, F, E> {
    fn eval<AP: CubicParser<E>>(&self, parser: &mut AP)
    where
        AP: AirParser<Field = F>,
    {
        let beta = self.challenge.eval(parser);

        let mut prev = parser.zero_extension();
        for (chunk, row_acc) in self.values.chunks_exact(2).zip(self.row_accumulators) {
            let a = chunk[0].eval_cubic(parser);
            let b = chunk[1].eval_cubic(parser);
            let acc = row_acc.eval(parser);
            let beta_minus_a = parser.sub_extension(beta, a);
            let beta_minus_b = parser.sub_extension(beta, b);
            let acc_minus_prev = parser.sub_extension(acc, prev);
            let mut product = parser.mul_extension(beta_minus_a, beta_minus_b);
            product = parser.mul_extension(product, acc_minus_prev);
            let mut constraint = parser.add_extension(beta_minus_a, beta_minus_b);
            constraint = parser.sub_extension(constraint, product);
            parser.constraint_extension(constraint);
            prev = acc;
        }

        let log_lookup_accumulator = self.log_lookup_accumulator.eval(parser);
        let log_lookup_accumulator_next = self.log_lookup_accumulator.next().eval(parser);

        let row_acc_length = self.row_accumulators.len();
        let accumulated_value = self.row_accumulators.get(row_acc_length - 1).eval(parser);
        let accumulated_value_next = self
            .row_accumulators
            .get(row_acc_length - 1)
            .next()
            .eval(parser);
        let mut acc_transition_constraint =
            parser.sub_extension(log_lookup_accumulator_next, log_lookup_accumulator);
        acc_transition_constraint =
            parser.sub_extension(acc_transition_constraint, accumulated_value_next);
        parser.constraint_extension_transition(acc_transition_constraint);

        let acc_first_row_constraint =
            parser.sub_extension(log_lookup_accumulator, accumulated_value);
        parser.constraint_extension_first_row(acc_first_row_constraint);

        // Add digest constraint
        let lookup_digest = self.local_digest.eval(parser);
        let lookup_digest_constraint = parser.sub_extension(lookup_digest, log_lookup_accumulator);
        parser.constraint_extension_last_row(lookup_digest_constraint);
    }

    fn eval_global<AP: CubicParser<E>>(&self, parser: &mut AP)
    where
        AP: AirParser<Field = F>,
    {
        let beta = self.challenge.eval(parser);

        // Constrain the public accumulation
        let mut prev = parser.zero_extension();
        for (chunk, row_acc) in self
            .public_values
            .chunks_exact(2)
            .zip(self.global_accumulators)
        {
            let a = chunk[0].eval_cubic(parser);
            let b = chunk[1].eval_cubic(parser);
            let acc = row_acc.eval(parser);
            let beta_minus_a = parser.sub_extension(beta, a);
            let beta_minus_b = parser.sub_extension(beta, b);
            let acc_minus_prev = parser.sub_extension(acc, prev);
            let mut product = parser.mul_extension(beta_minus_a, beta_minus_b);
            product = parser.mul_extension(product, acc_minus_prev);
            let mut constraint = parser.add_extension(beta_minus_a, beta_minus_b);
            constraint = parser.sub_extension(constraint, product);
            parser.constraint_extension(constraint);
            prev = acc;
        }
        let lookup_total_value = prev;

        // Add digest constraint
        if let Some(digest) = self.global_digest {
            let lookup_digest = digest.eval(parser);
            let lookup_digest_constraint = parser.sub_extension(lookup_digest, lookup_total_value);
            parser.constraint_extension_last_row(lookup_digest_constraint);
        }
    }

    pub(crate) fn digest_constraint(&self) -> LookupConstraint<T, F, E> {
        LookupConstraint::ValuesDigest(self.digest, self.local_digest, self.global_digest)
    }
}
