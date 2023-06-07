use super::{FpAddInstruction, FpInnerProductInstruction, FpMulConstInstruction, FpMulInstruction};
use crate::curta::air::parser::AirParser;
use crate::curta::bool::SelectInstruction;
use crate::curta::field::FpDenInstruction;
use crate::curta::instruction::{Extendable, Instruction, RichField};
use crate::curta::parameters::FieldParameters;
use crate::curta::register::{FieldRegister, MemorySlice};

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
