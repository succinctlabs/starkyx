//! Range checks based on logarithmic derivatives.
//!

pub mod log_der;

use log_der::LogLookup;
use plonky2::{field::{extension::{Extendable, FieldExtension}, packed::PackedField}, hash::hash_types::RichField, plonk::circuit_builder::CircuitBuilder};

use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

use super::chip::StarkParameters;

#[derive(Debug, Clone)]
pub enum Lookup {
    LogDerivative(LogLookup),
}


impl Lookup {
    pub fn packed_generic_constraints<
        L: StarkParameters<F, D>,
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>, 
        {
            match self {
                Lookup::LogDerivative(log_lookup) => {
                    log_lookup.packed_generic_constraints::<L, F, D, FE, P, D2, COLUMNS, PUBLIC_INPUTS>(vars, yield_constr)
                }
            }
        }

        pub fn ext_circuit_constraints<
        L: StarkParameters<F, D>,
        F: RichField + Extendable<D>,
        const D: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            Lookup::LogDerivative(log_lookup) => {
                log_lookup.ext_circuit_constraints::<L, F, D, COLUMNS, PUBLIC_INPUTS>(builder, vars, yield_constr)
            }
        }
    }
}