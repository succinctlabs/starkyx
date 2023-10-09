use super::LogLookupValues;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::kind::ConstraintKind;
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

        let log_lookup_accumulator = self.log_lookup_accumulator.eval(parser);
        let log_lookup_accumulator_next = self.log_lookup_accumulator.next().eval(parser);

        let acc_result = parser.sub_extension(log_lookup_accumulator_next, log_lookup_accumulator);

        let acc_result_constraint = LogConstraints::log_row_accumulation(
            parser,
            beta,
            &self.trace_values,
            self.row_accumulators,
            acc_result,
            ConstraintKind::Transition,
        );
        parser.constraint_extension_transition(acc_result_constraint);

        let acc_first_row_constraint = LogConstraints::log_row_accumulation(
            parser,
            beta,
            &self.trace_values,
            self.row_accumulators,
            acc_result,
            ConstraintKind::First,
        );
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

        if let Some(digest) = self.global_digest {
            let global_digest_constraint = LogConstraints::log_row_accumulation(
                parser,
                beta,
                &self.public_values,
                self.global_accumulators,
                digest.eval(parser),
                ConstraintKind::Global,
            );
            parser.constraint_extension_last_row(global_digest_constraint);
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
