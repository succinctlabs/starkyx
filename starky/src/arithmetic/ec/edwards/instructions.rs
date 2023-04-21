use den::Den;

use super::add::FromEdwardsAdd;
use super::*;
use crate::arithmetic::bool::Selector;
use crate::arithmetic::field::{
    FpAddInstruction, FpInnerProductInstruction, FpMulConstInstruction, FpMulInstruction,
};
use crate::arithmetic::instruction::Instruction;
use crate::arithmetic::register::MemorySlice;

#[derive(Debug, Clone)]
pub enum EdWardsMicroInstruction<E: EdwardsParameters> {
    Den(Den<E::FieldParam>),
    FpAdd(FpAddInstruction<E::FieldParam>),
    FpMul(FpMulInstruction<E::FieldParam>),
    FpInnerProduct(FpInnerProductInstruction<E::FieldParam>),
    FpMulConst(FpMulConstInstruction<E::FieldParam>),
    Selector(Selector<FieldRegister<E::FieldParam>>),
}

impl<E: EdwardsParameters, F: RichField + Extendable<D>, const D: usize> Instruction<F, D>
    for EdWardsMicroInstruction<E>
{
    fn witness_layout(&self) -> Vec<MemorySlice> {
        match self {
            EdWardsMicroInstruction::Den(den) => {
                <Den<E::FieldParam> as Instruction<F, D>>::witness_layout(den)
            }
            EdWardsMicroInstruction::FpAdd(fp_add) => {
                <FpAddInstruction<E::FieldParam> as Instruction<F, D>>::witness_layout(fp_add)
            }
            EdWardsMicroInstruction::FpMul(fp_mul) => {
                <FpMulInstruction<E::FieldParam> as Instruction<F, D>>::witness_layout(fp_mul)
            }
            EdWardsMicroInstruction::FpInnerProduct(fp_quad) => {
                <FpInnerProductInstruction<E::FieldParam> as Instruction<F, D>>::witness_layout(
                    fp_quad,
                )
            }
            EdWardsMicroInstruction::FpMulConst(fp_mul_const) => {
                <FpMulConstInstruction<E::FieldParam> as Instruction<F, D>>::witness_layout(
                    fp_mul_const,
                )
            }
            EdWardsMicroInstruction::Selector(selector) => {
                <Selector<FieldRegister<E::FieldParam>> as Instruction<F, D>>::witness_layout(
                    selector,
                )
            }
        }
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            EdWardsMicroInstruction::Den(den) => {
                <Den<E::FieldParam> as Instruction<F, D>>::assign_row(
                    den, trace_rows, row, row_index,
                )
            }
            EdWardsMicroInstruction::FpAdd(fp_add) => {
                <FpAddInstruction<E::FieldParam> as Instruction<F, D>>::assign_row(
                    fp_add, trace_rows, row, row_index,
                )
            }
            EdWardsMicroInstruction::FpMul(fp_mul) => {
                <FpMulInstruction<E::FieldParam> as Instruction<F, D>>::assign_row(
                    fp_mul, trace_rows, row, row_index,
                )
            }
            EdWardsMicroInstruction::FpInnerProduct(fp_quad) => {
                <FpInnerProductInstruction<E::FieldParam> as Instruction<F, D>>::assign_row(
                    fp_quad, trace_rows, row, row_index,
                )
            }
            EdWardsMicroInstruction::FpMulConst(fp_mul_const) => {
                <FpMulConstInstruction<E::FieldParam> as Instruction<F, D>>::assign_row(
                    fp_mul_const,
                    trace_rows,
                    row,
                    row_index,
                )
            }
            EdWardsMicroInstruction::Selector(selector) => {
                <Selector<FieldRegister<E::FieldParam>> as Instruction<F, D>>::assign_row(
                    selector, trace_rows, row, row_index,
                )
            }
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
        vars: crate::vars::StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        P: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        match self {
            EdWardsMicroInstruction::Den(den) => {
                <Den<E::FieldParam> as Instruction<F, D>>::packed_generic_constraints(
                    den,
                    vars,
                    yield_constr,
                )
            }
            EdWardsMicroInstruction::FpAdd(fp_add) => <FpAddInstruction<E::FieldParam> as Instruction<
                F,
                D,
            >>::packed_generic_constraints(
                fp_add, vars, yield_constr
            ),
            EdWardsMicroInstruction::FpMul(fp_mul) => <FpMulInstruction<E::FieldParam> as Instruction<
                F,
                D,
            >>::packed_generic_constraints(
                fp_mul, vars, yield_constr
            ),
            EdWardsMicroInstruction::FpInnerProduct(fp_quad) => <FpInnerProductInstruction<E::FieldParam> as Instruction<
                F,
                D,
            >>::packed_generic_constraints(
                fp_quad, vars, yield_constr
            ),
            EdWardsMicroInstruction::FpMulConst(fp_mul_const) => {
                <FpMulConstInstruction<E::FieldParam> as Instruction<F, D>>::packed_generic_constraints(
                    fp_mul_const,
                    vars,
                    yield_constr,
                )
            }
            EdWardsMicroInstruction::Selector(selector) => <Selector<
                FieldRegister<E::FieldParam>,
            > as Instruction<F, D>>::packed_generic_constraints(
                selector, vars, yield_constr
            ),
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: crate::vars::StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            EdWardsMicroInstruction::Den(den) => {
                <Den<E::FieldParam> as Instruction<F, D>>::ext_circuit_constraints(
                    den,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            EdWardsMicroInstruction::FpAdd(fp_add) => <FpAddInstruction<E::FieldParam> as Instruction<
                F,
                D,
            >>::ext_circuit_constraints(
                fp_add, builder, vars, yield_constr
            ),
            EdWardsMicroInstruction::FpMul(fp_mul) => <FpMulInstruction<E::FieldParam> as Instruction<
                F,
                D,
            >>::ext_circuit_constraints(
                fp_mul, builder, vars, yield_constr
            ),
            EdWardsMicroInstruction::FpInnerProduct(fp_quad) => <FpInnerProductInstruction<E::FieldParam> as Instruction<
                F,
                D,
            >>::ext_circuit_constraints(
                fp_quad, builder, vars, yield_constr
            ),
            EdWardsMicroInstruction::FpMulConst(fp_mul_const) => {
                <FpMulConstInstruction<E::FieldParam> as Instruction<F, D>>::ext_circuit_constraints(
                    fp_mul_const,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            EdWardsMicroInstruction::Selector(selector) => <Selector<
                FieldRegister<E::FieldParam>,
            > as Instruction<F, D>>::ext_circuit_constraints(
                selector, builder, vars, yield_constr
            ),
        }
    }
}

impl<E: EdwardsParameters> From<FpMulInstruction<E::FieldParam>> for EdWardsMicroInstruction<E> {
    fn from(fp_mul: FpMulInstruction<E::FieldParam>) -> Self {
        EdWardsMicroInstruction::FpMul(fp_mul)
    }
}

impl<E: EdwardsParameters> From<FpAddInstruction<E::FieldParam>> for EdWardsMicroInstruction<E> {
    fn from(fp_add: FpAddInstruction<E::FieldParam>) -> Self {
        EdWardsMicroInstruction::FpAdd(fp_add)
    }
}

impl<E: EdwardsParameters> From<FpInnerProductInstruction<E::FieldParam>>
    for EdWardsMicroInstruction<E>
{
    fn from(fp_quad: FpInnerProductInstruction<E::FieldParam>) -> Self {
        EdWardsMicroInstruction::FpInnerProduct(fp_quad)
    }
}

impl<E: EdwardsParameters> From<FpMulConstInstruction<E::FieldParam>>
    for EdWardsMicroInstruction<E>
{
    fn from(fp_mul_const: FpMulConstInstruction<E::FieldParam>) -> Self {
        EdWardsMicroInstruction::FpMulConst(fp_mul_const)
    }
}

impl<E: EdwardsParameters> From<Den<E::FieldParam>> for EdWardsMicroInstruction<E> {
    fn from(den: Den<E::FieldParam>) -> Self {
        EdWardsMicroInstruction::Den(den)
    }
}

impl<E: EdwardsParameters> From<Selector<FieldRegister<E::FieldParam>>>
    for EdWardsMicroInstruction<E>
{
    fn from(selector: Selector<FieldRegister<E::FieldParam>>) -> Self {
        EdWardsMicroInstruction::Selector(selector)
    }
}

impl<E: EdwardsParameters> FromEdwardsAdd<E> for EdWardsMicroInstruction<E> {}
