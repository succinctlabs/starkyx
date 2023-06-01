use plonky2::iop::ext_target::ExtensionTarget;

use super::{FpAddInstruction, FpInnerProductInstruction, FpMulConstInstruction, FpMulInstruction};
use crate::curta::bool::SelectInstruction;
use crate::curta::field::FpDenInstruction;
use crate::curta::instruction::{
    CircuitBuilder, Extendable, FieldExtension, Instruction, PackedField, RichField,
};
use crate::curta::parameters::FieldParameters;
use crate::curta::register::{FieldRegister, MemorySlice};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use crate::curta::new_stark::vars as new_vars;
use crate::curta::air::parser::AirParser;

macro_rules! instruction_set {
    ($($var:ident$(<$($t:ident),+>)?),*) => {

        #[derive(Debug)]
        pub enum InstructionSet<P: FieldParameters> { $(
            $var($var$(< $( $t ),+ >)?),
        )*}

        impl<P: FieldParameters> Clone for InstructionSet<P> {
            fn clone(&self) -> Self {
                match &self {
                    $(InstructionSet::$var(variant) => InstructionSet::$var(variant.clone()),)*
                }
            }
        }

        $(
            impl<P: FieldParameters> From<$var$(< $( $t ),+ >)?> for InstructionSet<P> {
                fn from(x: $var$(< $( $t ),+ >)?) -> Self {
                    InstructionSet::$var(x)
                }
            }
        )*

        impl<P: FieldParameters, F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for InstructionSet<P> {
            fn trace_layout(&self) -> Vec<MemorySlice> {
                match &self {
                    $(InstructionSet::$var(variant) => Instruction::<F, D>::trace_layout(variant),)*
                }
            }

            fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
                match &self {
                    $(InstructionSet::$var(variant) => Instruction::<F, D>::assign_row(variant, trace_rows, row, row_index),)*
                }
            }

            fn packed_generic<
                FE,
                Q,
                const D2: usize,
                const COLUMNS: usize,
                const PUBLIC_INPUTS: usize,
            >(
                &self,
                vars: StarkEvaluationVars<FE, Q, { COLUMNS }, { PUBLIC_INPUTS }>,
            ) -> Vec<Q> where
                FE: FieldExtension<D2, BaseField = F>,
                Q: PackedField<Scalar = FE>
            {
                match &self {
                    $(InstructionSet::$var(variant) => Instruction::<F, D>::packed_generic(variant, vars),)*
                }
            }

            fn ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
                &self,
                builder: &mut CircuitBuilder<F, D>,
                vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
            ) -> Vec<ExtensionTarget<D>> {
                match &self {
                    $(InstructionSet::$var(variant) => Instruction::<F, D>::ext_circuit(variant, builder, vars),)*
                }
            }

            fn packed_generic_new<
                FE,
                Q,
                const D2: usize,
                const COLUMNS: usize,
                const PUBLIC_INPUTS: usize,
                const CHALLENGES: usize,
            >(
                &self,
                vars: new_vars::StarkEvaluationVars<FE, Q, { COLUMNS }, { PUBLIC_INPUTS }, {CHALLENGES}>,
            ) -> Vec<Q> where
                FE: FieldExtension<D2, BaseField = F>,
                Q: PackedField<Scalar = FE>
            {
                match &self {
                    $(InstructionSet::$var(variant) => Instruction::<F, D>::packed_generic_new(variant, vars),)*
                }
            }

            fn ext_circuit_new<
            const COLUMNS: usize,
            const PUBLIC_INPUTS: usize,
            const CHALLENGES: usize,
        >(
            &self,
            builder: &mut CircuitBuilder<F, D>,
            vars: new_vars::StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
        ) -> Vec<ExtensionTarget<D>> {
            match &self {
                $(InstructionSet::$var(variant) => Instruction::<F, D>::ext_circuit_new(variant, builder, vars),)*
            } 
        }

            fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var> {
                match &self {
                    $(InstructionSet::$var(variant) => Instruction::<F, D>::eval(variant, parser),)*
                }
            }
        }
    }

}

type FieldSelectorInstruction<P> = SelectInstruction<FieldRegister<P>>;

instruction_set! {
    FpAddInstruction<P>,
    FpMulInstruction<P>,
    FpMulConstInstruction<P>,
    FpInnerProductInstruction<P>,
    FpDenInstruction<P>,
    FieldSelectorInstruction<P>
}

pub trait FromInstructionSet<P: FieldParameters>:
    From<FpAddInstruction<P>>
    + From<FpMulInstruction<P>>
    + From<FpMulConstInstruction<P>>
    + From<FpInnerProductInstruction<P>>
    + From<FpDenInstruction<P>>
    + From<FieldSelectorInstruction<P>>
{
}

impl<P: FieldParameters> FromInstructionSet<P> for InstructionSet<P> {}
