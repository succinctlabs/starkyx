//! Range checks based on logarithmic derivatives.
//!

pub mod log_der;

use log_der::LogLookup;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone)]
pub enum Lookup {
    LogDerivative(LogLookup),
}

impl Lookup {
    pub fn packed_generic_constraints<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        betas: &[[F; 3]],
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            Lookup::LogDerivative(log_lookup) => log_lookup
                .packed_generic_constraints::<F, D, FE, P, D2, COLUMNS, PUBLIC_INPUTS>(
                    betas,
                    vars,
                    yield_constr,
                ),
        }
    }

    pub fn ext_circuit_constraints<
        F: RichField + Extendable<D>,
        const D: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        betas: &[[F; 3]],
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            Lookup::LogDerivative(log_lookup) => log_lookup
                .ext_circuit_constraints::<F, D, COLUMNS, PUBLIC_INPUTS>(
                    betas,
                    builder,
                    vars,
                    yield_constr,
                ),
        }
    }
}
