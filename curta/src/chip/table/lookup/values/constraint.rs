use itertools::Itertools;

use super::LogLookupValues;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::Constraint;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::log_derivative::constraints::LogConstraints;
use crate::chip::table::lookup::constraint::LookupConstraint;
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
            let a = chunk[0].eval(parser);
            let b = chunk[1].eval(parser);
            let acc = row_acc.eval(parser);
            let acc_minus_prev = parser.sub_extension(acc, prev);
            let constraint = LogConstraints::log_arithmetic(parser, beta, a, b, acc_minus_prev);
            parser.constraint_extension(constraint);
            prev = acc;
        }

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
        if let Some(last) = last_element {
            let a = last.next().eval(parser);
            acc_transition_constraint =
                LogConstraints::log(parser, beta, a, acc_transition_constraint);
        }
        parser.constraint_extension_transition(acc_transition_constraint);

        let mut acc_first_row_constraint =
            parser.sub_extension(log_lookup_accumulator, accumulated_value);
        if let Some(last) = last_element {
            let a = last.eval(parser);
            acc_first_row_constraint =
                LogConstraints::log(parser, beta, a, acc_first_row_constraint);
        }
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
            let a = chunk[0].eval(parser);
            let b = chunk[1].eval(parser);
            let acc = row_acc.eval(parser);
            let acc_minus_prev = parser.sub_extension(acc, prev);
            let constraint = LogConstraints::log_arithmetic(parser, beta, a, b, acc_minus_prev);
            parser.constraint_extension(constraint);
            prev = acc;
        }

        let lookup_total_value = prev;

        // Add digest constraint
        if let Some(digest) = self.global_digest {
            let lookup_digest = digest.eval(parser);
            let mut lookup_digest_constraint =
                parser.sub_extension(lookup_digest, lookup_total_value);
            if let Some(last) = last_element {
                let a = last.eval(parser);
                lookup_digest_constraint =
                    LogConstraints::log(parser, beta, a, lookup_digest_constraint);
            }
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
