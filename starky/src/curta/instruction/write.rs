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

    fn packed_generic<FE, P, const D2: usize, const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        _vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<P>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        vec![]
    }

    fn ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<ExtensionTarget<D>> {
        vec![]
    }
}
