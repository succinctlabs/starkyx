use super::LogLookupValues;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::lookup::log_der::constraint::LookupConstraint;
use crate::math::prelude::*;

impl<T: EvalCubic, F: Field, E: CubicParameters<F>> LogLookupValues<T, F, E> {
    pub(crate) fn eval<AP: CubicParser<E>>(&self, parser: &mut AP)
    where
        AP: AirParser<Field = F>,
    {
        let beta = self.challenge.eval(parser);

        let mut prev = parser.zero_extension();
        for (chunk, row_acc) in self.trace_values.chunks_exact(2).zip(self.row_accumulators) {
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

    pub(crate) fn eval_global<AP: CubicParser<E>>(&self, parser: &mut AP)
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
