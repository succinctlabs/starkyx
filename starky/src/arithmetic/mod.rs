#![allow(dead_code)]

pub mod add;
pub mod arithmetic_stark;
pub mod eddsa;
pub mod mul;
pub mod polynomial;
pub(crate) mod util;

use std::sync::mpsc::Sender;

use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_maybe_rayon::*;

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
    pub const fn index(&self) -> usize {
        match self {
            Register::Local(index, _) => *index,
            Register::Next(index, _) => *index,
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

    #[inline]
    pub fn assign<T: Copy>(&self, trace_rows: &mut [Vec<T>], value: &mut [T], row_index: usize) {
        match self {
            Register::Local(index, length) => {
                trace_rows[row_index][*index..*index + length].copy_from_slice(value);
            }
            Register::Next(index, length) => {
                trace_rows[row_index + 1][*index..*index + length].copy_from_slice(value);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ArithmeticOp {
    AddMod(BigUint, BigUint, BigUint),
    MulMod(BigUint, BigUint, BigUint),
    EdAdd(BigUint, BigUint),
}

#[derive(Debug, Clone)]
pub enum ArithmeticLayout {
    Add(AddModLayout),
    Mul(MulModLayout),
}

impl ArithmeticLayout {
    pub fn allocation_registers(&self) -> (Register, Register, Register) {
        match self {
            ArithmeticLayout::Add(layout) => layout.allocation_registers(),
            ArithmeticLayout::Mul(layout) => layout.allocation_registers(),
        }
    }

    pub fn assign_row<T: Copy>(&self, trace_rows: &mut [Vec<T>], row: &mut [T], row_index: usize) {
        match self {
            ArithmeticLayout::Add(layout) => layout.assign_row(trace_rows, row, row_index),
            ArithmeticLayout::Mul(layout) => layout.assign_row(trace_rows, row, row_index),
        }
    }
}

/// An experimental parser to generate Stark constaint code from commands
///
/// The output is writing to a "memory" passed to it.
#[derive(Debug, Clone, Copy)]
pub struct ArithmeticParser<F, const D: usize> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    pub fn op_trace_row(
        row: usize,
        op_index: usize,
        tx: Sender<(usize, usize, Vec<F>)>,
        operation: ArithmeticOp,
    ) {
        match operation {
            ArithmeticOp::AddMod(a, b, m) => rayon::spawn(move || {
                let mut row_vec = Self::add_trace(a, b, m);
                row_vec.push(F::from_canonical_usize(row));
                tx.send((row, op_index, row_vec)).unwrap()
            }),
            ArithmeticOp::MulMod(a, b, m) => rayon::spawn(move || {
                let mut row_vec = Self::mul_trace(a, b, m);
                row_vec.push(F::from_canonical_usize(row));
                tx.send((row, op_index, row_vec)).unwrap()
            }),
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
