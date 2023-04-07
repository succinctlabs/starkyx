pub mod add;
pub mod mul;

pub use add::AddModLayout;
pub use mul::MulModLayout;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::layout::OpcodeLayout;
use super::*;
use crate::arithmetic::layout::Opcode;

#[derive(Debug, Clone)]
pub enum ArithmeticLayout {
    Add(AddModLayout),
    Mul(MulModLayout),
}

#[derive(Debug, Clone)]
pub enum ArithmeticOp {
    AddMod(BigUint, BigUint, BigUint),
    MulMod(BigUint, BigUint, BigUint),
}

impl<F: RichField + Extendable<D>, const D: usize> OpcodeLayout<F, D> for ArithmeticLayout {
    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            ArithmeticLayout::Add(layout) => layout.assign_row(trace_rows, row, row_index),
            ArithmeticLayout::Mul(layout) => layout.assign_row(trace_rows, row, row_index),
        }
    }

    fn packed_generic_constraints<
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
            ArithmeticLayout::Add(layout) => {
                ArithmeticParser::add_packed_generic_constraints(*layout, vars, yield_constr)
            }
            ArithmeticLayout::Mul(layout) => {
                ArithmeticParser::mul_packed_generic_constraints(*layout, vars, yield_constr)
            }
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            ArithmeticLayout::Add(layout) => {
                ArithmeticParser::add_ext_circuit(*layout, builder, vars, yield_constr)
            }
            ArithmeticLayout::Mul(layout) => {
                ArithmeticParser::mul_ext_circuit(*layout, builder, vars, yield_constr)
            }
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Opcode<F, D> for ArithmeticOp {
    type Output = ();
    fn generate_trace_row(self) -> (Vec<F>, ()) {
        let trace_row = match self {
            ArithmeticOp::AddMod(a, b, m) => ArithmeticParser::add_trace(a, b, m),
            ArithmeticOp::MulMod(a, b, m) => ArithmeticParser::mul_trace(a, b, m),
        };
        (trace_row, ())
    }
}
