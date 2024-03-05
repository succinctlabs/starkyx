use super::LogLookupValues;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::Constraint;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::table::log_derivative::constraints::LogConstraints;
use crate::chip::table::lookup::constraint::LookupConstraint;
use crate::chip::AirParameters;
use crate::math::prelude::*;

impl<T: EvalCubic, F: Field, E: CubicParameters<F>> LogLookupValues<T, F, E> {
    pub(crate) fn eval<AP>(&self, parser: &mut AP)
    where
        AP: CubicParser<E>,
        AP: AirParser<Field = F>,
    {
        let beta = self.challenge.eval(parser);

        LogConstraints::log_trace_accumulation(
            parser,
            beta,
            &self.trace_values,
            &self.row_accumulators,
            self.local_digest,
        );
    }

    pub(crate) fn eval_global<AP>(&self, parser: &mut AP)
    where
        AP: CubicParser<E>,
        AP: AirParser<Field = F>,
    {
        let beta = self.challenge.eval(parser);

        if let Some(digest) = self.global_digest {
            LogConstraints::log_global_accumulation(
                parser,
                beta,
                &self.public_values,
                &self.global_accumulators,
                digest,
            );
        }
    }
}

impl<F: Field, E: CubicParameters<F>> LogLookupValues<ElementRegister, F, E> {
    pub(crate) fn register_constraints<L: AirParameters<Field = F, CubicParams = E>>(
        &self,
        builder: &mut AirBuilder<L>,
    ) {
        // Register the constraints on the trace values.
        builder.constraints.push(Constraint::lookup(
            LookupConstraint::<ElementRegister, _, _>::ValuesLocal(self.clone()).into(),
        ));
        // If global values are present, register the constraints on the global values.
        if self.global_digest.is_some() {
            builder.global_constraints.push(Constraint::lookup(
                LookupConstraint::<ElementRegister, _, _>::ValuesGlobal(self.clone()).into(),
            ));
        }
        // Register the constraints on the digest.
        builder.constraints.push(Constraint::lookup(
            LookupConstraint::<ElementRegister, _, _>::ValuesDigest(
                self.digest,
                self.local_digest,
                self.global_digest,
            )
            .into(),
        ));
    }
}

impl<F: Field, E: CubicParameters<F>> LogLookupValues<CubicRegister, F, E> {
    pub(crate) fn register_constraints<L: AirParameters<Field = F, CubicParams = E>>(
        &self,
        builder: &mut AirBuilder<L>,
    ) {
        // Register the constraints on the trace values.
        builder.constraints.push(Constraint::lookup(
            LookupConstraint::<CubicRegister, _, _>::ValuesLocal(self.clone()).into(),
        ));
        // If global values are present, register the constraints on the global values.
        if self.global_digest.is_some() {
            builder.global_constraints.push(Constraint::lookup(
                LookupConstraint::<CubicRegister, _, _>::ValuesGlobal(self.clone()).into(),
            ));
        }
        // Register the constraints on the digest.
        builder.constraints.push(Constraint::lookup(
            LookupConstraint::<CubicRegister, _, _>::ValuesDigest(
                self.digest,
                self.local_digest,
                self.global_digest,
            )
            .into(),
        ));
    }
}
