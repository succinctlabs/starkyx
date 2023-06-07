use super::*;

#[derive(Clone, Debug, Copy)]
pub struct WriteInstruction(pub MemorySlice);

impl WriteInstruction {
    #[inline]
    pub fn into_register(self) -> MemorySlice {
        self.0
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for WriteInstruction {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![self.0]
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.0.assign(trace_rows, 0, row, row_index);
    }

    fn eval<AP: AirParser<Field = F>>(&self, _parser: &mut AP) -> Vec<AP::Var> {
        vec![]
    }
}
