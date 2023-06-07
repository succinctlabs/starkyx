use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::Instruction;
use crate::curta::register::MemorySlice;

/// A defult instruction set that contains no custom instructions
#[derive(Clone, Debug)]
pub struct EmptyInstructionSet<F, const D: usize> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for EmptyInstructionSet<F, D> {
    fn assign_row(&self, _trace_rows: &mut [Vec<F>], _row: &mut [F], _row_index: usize) {}

    fn trace_layout(&self) -> Vec<MemorySlice> {
        Vec::new()
    }

    fn eval<AP: crate::curta::air::parser::AirParser<Field = F>>(
        &self,
        _parser: &mut AP,
    ) -> Vec<AP::Var> {
        vec![]
    }
}
