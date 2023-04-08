use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::register::{DataRegister, WitnessData};
use super::Register;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Clone, Debug, Hash, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionID(pub [char; 6]);

#[derive(Clone, Debug)]
pub struct LabeledInstruction<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> {
    label: InstructionID,
    instruction: I,
    _marker: core::marker::PhantomData<F>,
}

pub trait Instruction<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Clone
{
    //fn generate_trace_row(&self, input: Option<Self::Input>) -> (Vec<F>, Option<Self::Output>);

    fn shift_right(&mut self, free_shift: usize, arithmetic_shift: usize);

    fn witness_data(&self) -> WitnessData;

    fn set_witness(&mut self, witness: Register) -> Result<()>;

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize);

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
        P: PackedField<Scalar = FE>;

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    );
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize>
    LabeledInstruction<I, F, D>
{
    pub fn new(label: InstructionID, instruction: I) -> Self {
        Self {
            label,
            instruction,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn destruct(self) -> (InstructionID, I) {
        (self.label, self.instruction)
    }
}
