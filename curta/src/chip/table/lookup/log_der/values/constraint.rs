use itertools::Itertools;

use super::LogLookupValues;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::Constraint;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::lookup::log_der::constraint::LookupConstraint;
use crate::chip::AirParameters;
use crate::math::prelude::*;

impl<T: EvalCubic, F: Field, E: CubicParameters<F>> LogLookupValues<T, F, E> {
    pub(crate) fn eval<AP: CubicParser<E>>(&self, parser: &mut AP)
    where
        AP: AirParser<Field = F>,
    {
        let beta = self.challenge.eval(parser);

        let value_chunks = self.trace_values.chunks_exact(2);
        let last_element = value_chunks.remainder().first();

        let zero = parser.zero_extension();

        let mut prev = zero;
        for (chunk, row_acc) in value_chunks.zip_eq(self.row_accumulators) {
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

        let (acc_trans_rem, acc_trans_mult) = match last_element {
            Some(a_reg) => {
                let a = a_reg.next().eval_cubic(parser);
                let beta_minus_a = parser.sub_extension(beta, a);
                (parser.one_extension(), beta_minus_a)
            }
            None => (zero, parser.one_extension()),
        };

        let (acc_first_rem, acc_first_mult) = match last_element {
            Some(a_reg) => {
                let a = a_reg.eval_cubic(parser);
                let beta_minus_a = parser.sub_extension(beta, a);
                (parser.one_extension(), beta_minus_a)
            }
            None => (zero, parser.one_extension()),
        };

        let log_lookup_accumulator = self.log_lookup_accumulator.eval(parser);
        let log_lookup_accumulator_next = self.log_lookup_accumulator.next().eval(parser);

        let accumulated_value = self
            .row_accumulators
            .last()
            .map_or(zero, |r| r.eval(parser));
        let accumulated_value_next = self
            .row_accumulators
            .last()
            .map_or(zero, |r| r.next().eval(parser));
        let mut acc_transition_constraint =
            parser.sub_extension(log_lookup_accumulator_next, log_lookup_accumulator);
        acc_transition_constraint =
            parser.sub_extension(acc_transition_constraint, accumulated_value_next);
        acc_transition_constraint = parser.mul_extension(acc_transition_constraint, acc_trans_mult);
        acc_transition_constraint = parser.sub_extension(acc_transition_constraint, acc_trans_rem);
        parser.constraint_extension_transition(acc_transition_constraint);

        let mut acc_first_row_constraint =
            parser.sub_extension(log_lookup_accumulator, accumulated_value);
        acc_first_row_constraint = parser.mul_extension(acc_first_row_constraint, acc_first_mult);
        acc_first_row_constraint = parser.sub_extension(acc_first_row_constraint, acc_first_rem);
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

        let value_chunks = self.public_values.chunks_exact(2);
        let last_element = value_chunks.remainder().first();

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

        let (acc_rem, acc_mult) = match last_element {
            Some(a_reg) => {
                let a = a_reg.eval_cubic(parser);
                let beta_minus_a = parser.sub_extension(beta, a);
                (parser.one_extension(), beta_minus_a)
            }
            None => (parser.zero_extension(), parser.one_extension()),
        };

        // Add digest constraint
        if let Some(digest) = self.global_digest {
            let lookup_digest = digest.eval(parser);
            let mut lookup_digest_constraint =
                parser.sub_extension(lookup_digest, lookup_total_value);
            lookup_digest_constraint = parser.mul_extension(lookup_digest_constraint, acc_mult);
            lookup_digest_constraint = parser.sub_extension(lookup_digest_constraint, acc_rem);
            parser.constraint_extension_last_row(lookup_digest_constraint);
        }
    }
}

impl<F: Field, E: CubicParameters<F>> LogLookupValues<ElementRegister, F, E> {
    pub(crate) fn register_constraints<L: AirParameters<Field = F, CubicParams = E>>(
        &self,
        builder: &mut AirBuilder<L>,
    ) {
        builder.constraints.push(Constraint::lookup(
            LookupConstraint::<ElementRegister, _, _>::ValuesLocal(self.clone()).into(),
        ));
        if self.global_digest.is_some() {
            builder.global_constraints.push(Constraint::lookup(
                LookupConstraint::<ElementRegister, _, _>::ValuesGlobal(self.clone()).into(),
            ));
        }
    }
}

impl<F: Field, E: CubicParameters<F>> LogLookupValues<CubicRegister, F, E> {
    pub(crate) fn register_constraints<L: AirParameters<Field = F, CubicParams = E>>(
        &self,
        builder: &mut AirBuilder<L>,
    ) {
        builder.constraints.push(Constraint::lookup(
            LookupConstraint::<CubicRegister, _, _>::ValuesLocal(self.clone()).into(),
        ));
        if self.global_digest.is_some() {
            builder.global_constraints.push(Constraint::lookup(
                LookupConstraint::<CubicRegister, _, _>::ValuesGlobal(self.clone()).into(),
            ));
        }
    }
}
