#![allow(dead_code)]

pub mod add;
pub mod arithmetic_stark;
pub mod mul;
pub mod polynomial;
pub(crate) mod util;

use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use self::add::AddModLayout;
use self::mul::MulModLayout;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub enum Register {
    Local(usize, usize),
    Next(usize, usize),
}

impl Register {
    #[inline]
    pub const fn get_range(&self) -> (usize, usize) {
        match self {
            Register::Local(index, length) => (*index, *index + length),
            Register::Next(index, length) => (*index, *index + length),
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        match self {
            Register::Local(_, length) => *length,
            Register::Next(_, length) => *length,
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub enum ArithmeticOp {
    AddMod(BigUint, BigUint, BigUint),
    SubMod(BigUint, BigUint, BigUint),
    MulMod(BigUint, BigUint, BigUint),
}

pub enum ArithmeticLayout {
    Add(AddModLayout),
    Mul(MulModLayout),
}

/// An experimental parser to generate Stark constaint code from commands
///
/// The output is writing to a "memory" passed to it.
#[derive(Debug, Clone, Copy)]
pub struct ArithmeticParser<F, const D: usize> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    pub fn op_trace_row(operation: ArithmeticOp) -> Vec<F> {
        match operation {
            ArithmeticOp::AddMod(a, b, m) => Self::add_trace(a, b, m),
            ArithmeticOp::MulMod(a, b, m) => Self::mul_trace(a, b, m),
            _ => unimplemented!("Operation not supported"),
        }
    }

    pub fn op_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        layout: ArithmeticLayout,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match layout {
            ArithmeticLayout::Add(layout) => {
                Self::add_packed_generic_constraints(layout, vars, yield_constr)
            }
            ArithmeticLayout::Mul(layout) => {
                Self::mul_packed_generic_constraints(layout, vars, yield_constr)
            } //_ => unimplemented!("Operation not supported"),
        }
    }
    pub fn op_ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        layout: ArithmeticLayout,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match layout {
            ArithmeticLayout::Add(layout) => {
                Self::add_ext_circuit(layout, builder, vars, yield_constr)
            }
            ArithmeticLayout::Mul(layout) => {
                Self::mul_ext_circuit(layout, builder, vars, yield_constr)
            } //_ => unimplemented!("Operation not supported"),
        }
    }
}
