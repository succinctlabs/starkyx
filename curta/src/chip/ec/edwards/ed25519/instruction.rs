use serde::{Deserialize, Serialize};

use super::params::Ed25519BaseField;
use super::sqrt::Ed25519FpSqrtInstruction;
use crate::air::AirConstraint;
use crate::chip::field::add::FpAddInstruction;
use crate::chip::field::den::FpDenInstruction;
use crate::chip::field::div::FpDivInstruction;
use crate::chip::field::inner_product::FpInnerProductInstruction;
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::mul_const::FpMulConstInstruction;
use crate::chip::field::sub::FpSubInstruction;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::TraceWriter;
use crate::math::field::PrimeField64;
use crate::polynomial::parser::PolynomialParser;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum Ed25519FpInstruction {
    Add(FpAddInstruction<Ed25519BaseField>),
    Mul(FpMulInstruction<Ed25519BaseField>),
    MulConst(FpMulConstInstruction<Ed25519BaseField>),
    Inner(FpInnerProductInstruction<Ed25519BaseField>),
    Den(FpDenInstruction<Ed25519BaseField>),
    Sub(FpSubInstruction<Ed25519BaseField>),
    Div(FpDivInstruction<Ed25519BaseField>),
    Sqrt(Ed25519FpSqrtInstruction),
}

pub trait FromEd25519FieldInstruction:
    From<FpAddInstruction<Ed25519BaseField>>
    + From<FpMulInstruction<Ed25519BaseField>>
    + From<FpSubInstruction<Ed25519BaseField>>
    + From<FpDivInstruction<Ed25519BaseField>>
    + From<FpMulConstInstruction<Ed25519BaseField>>
    + From<FpInnerProductInstruction<Ed25519BaseField>>
    + From<FpDenInstruction<Ed25519BaseField>>
    + From<Ed25519FpSqrtInstruction>
{
}

impl FromEd25519FieldInstruction for Ed25519FpInstruction {}

impl<AP: PolynomialParser> AirConstraint<AP> for Ed25519FpInstruction {
    fn eval(&self, parser: &mut AP) {
        match self {
            Ed25519FpInstruction::Add(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
            Ed25519FpInstruction::Mul(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
            Ed25519FpInstruction::MulConst(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
            Ed25519FpInstruction::Inner(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
            Ed25519FpInstruction::Den(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
            Ed25519FpInstruction::Sub(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
            Ed25519FpInstruction::Div(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
            Ed25519FpInstruction::Sqrt(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
        }
    }
}

impl<F: PrimeField64> Instruction<F> for Ed25519FpInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Ed25519FpInstruction::Add(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::Mul(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::MulConst(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::Inner(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::Den(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::Sub(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::Div(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::Sqrt(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
        }
    }
}

impl From<FpAddInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(instr: FpAddInstruction<Ed25519BaseField>) -> Self {
        Ed25519FpInstruction::Add(instr)
    }
}

impl From<FpMulInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(instr: FpMulInstruction<Ed25519BaseField>) -> Self {
        Ed25519FpInstruction::Mul(instr)
    }
}

impl From<FpMulConstInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(instr: FpMulConstInstruction<Ed25519BaseField>) -> Self {
        Ed25519FpInstruction::MulConst(instr)
    }
}

impl From<FpInnerProductInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(instr: FpInnerProductInstruction<Ed25519BaseField>) -> Self {
        Ed25519FpInstruction::Inner(instr)
    }
}

impl From<FpDenInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(instr: FpDenInstruction<Ed25519BaseField>) -> Self {
        Ed25519FpInstruction::Den(instr)
    }
}

impl From<FpSubInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(instr: FpSubInstruction<Ed25519BaseField>) -> Self {
        Ed25519FpInstruction::Sub(instr)
    }
}

impl From<FpDivInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(instr: FpDivInstruction<Ed25519BaseField>) -> Self {
        Ed25519FpInstruction::Div(instr)
    }
}

impl From<Ed25519FpSqrtInstruction> for Ed25519FpInstruction {
    fn from(instr: Ed25519FpSqrtInstruction) -> Self {
        Ed25519FpInstruction::Sqrt(instr)
    }
}
