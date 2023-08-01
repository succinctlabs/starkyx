use self::and::And;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
pub use crate::math::prelude::*;

pub mod and;
pub mod or;
pub mod rotate;
pub mod shr;
pub mod xor;

#[derive(Debug, Clone, Copy)]
pub enum BitOperation<const NUM_BITS: usize> {
    And(And<NUM_BITS>),
}

impl<const NUM_BITS: usize> BitOperation<NUM_BITS> {
    pub fn a_bits(&self) -> ArrayRegister<BitRegister> {
        match self {
            BitOperation::And(and) => and.a,
        }
    }

    pub fn b_bits(&self) -> ArrayRegister<BitRegister> {
        match self {
            BitOperation::And(and) => and.b,
        }
    }

    pub fn result_bits(&self) -> ArrayRegister<BitRegister> {
        match self {
            BitOperation::And(and) => and.result,
        }
    }
}

impl<AP: AirParser, const NUM_BITS: usize> AirConstraint<AP> for BitOperation<NUM_BITS> {
    fn eval(&self, parser: &mut AP) {
        match self {
            BitOperation::And(and) => and.eval(parser),
        }
    }
}

impl<F: Field, const NUM_BITS: usize> Instruction<F> for BitOperation<NUM_BITS> {
    fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            BitOperation::And(and) => Instruction::<F>::inputs(and),
        }
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            BitOperation::And(and) => Instruction::<F>::trace_layout(and),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            BitOperation::And(and) => Instruction::<F>::write(and, writer, row_index),
        }
    }
}

impl<const N: usize> From<And<N>> for BitOperation<N> {
    fn from(and: And<N>) -> Self {
        BitOperation::And(and)
    }
}
