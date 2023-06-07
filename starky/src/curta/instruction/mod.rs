//! Instruction trait
//!
//! The instruction trait represents the interface of a microcomand in a chip. It is the
//! lowest level of abstraction in the arithmetic module.

pub mod empty;
mod set;
pub mod write;

pub use empty::EmptyInstructionSet;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
pub use set::{FromInstructionSet, InstructionSet};

use super::air::parser::AirParser;
use super::constraint::expression::ConstraintExpression;
use super::field::{
    FpAddInstruction, FpInnerProductInstruction, FpMulConstInstruction, FpMulInstruction,
};
use super::register::MemorySlice;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub trait Instruction<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Clone
{
    /// Returns a vector of memory slices or contiguous memory regions of the row in the trace that
    /// instruction relies on. These registers must be filled in by the `TraceWriter`.
    fn trace_layout(&self) -> Vec<MemorySlice>;

    /// Assigns the row in the trace according to the `witness_layout`. Usually called by the
    /// `TraceWriter`.
    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.trace_layout()
            .into_iter()
            .fold(0, |local_index, memory_slice| {
                memory_slice.assign(trace_rows, local_index, row, row_index)
            });
    }

    /// Returns evaluations of vanishing polynomials constraints.
    fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var>;

    fn constraint_degree() -> usize {
        2
    }

    fn expr(&self) -> ConstraintExpression<Self, F, D> {
        ConstraintExpression::Instruction(self.clone())
    }
}
