use serde::{Deserialize, Serialize};

use super::add::FpAddInstruction;
use super::den::FpDenInstruction;
use super::div::FpDivInstruction;
use super::inner_product::FpInnerProductInstruction;
use super::mul::FpMulInstruction;
use super::mul_const::FpMulConstInstruction;
use super::parameters::FieldParameters;
use super::register::FieldRegister;
use super::sub::FpSubInstruction;
use crate::air::AirConstraint;
use crate::chip::bool::SelectInstruction;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum FpInstruction<P: FieldParameters> {
    Add(FpAddInstruction<P>),
    Mul(FpMulInstruction<P>),
    MulConst(FpMulConstInstruction<P>),
    Inner(FpInnerProductInstruction<P>),
    Den(FpDenInstruction<P>),
    Select(SelectInstruction<FieldRegister<P>>),
    Sub(FpSubInstruction<P>),
    Div(FpDivInstruction<P>),
}

pub trait FromFieldInstruction<P: FieldParameters>:
    From<FpAddInstruction<P>>
    + From<FpMulInstruction<P>>
    + From<FpMulConstInstruction<P>>
    + From<FpInnerProductInstruction<P>>
    + From<FpDenInstruction<P>>
    + From<SelectInstruction<FieldRegister<P>>>
{
}

impl<P: FieldParameters> FromFieldInstruction<P> for FpInstruction<P> {}

impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        match self {
            FpInstruction::Add(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            FpInstruction::Mul(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            FpInstruction::MulConst(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            FpInstruction::Inner(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            FpInstruction::Den(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            FpInstruction::Select(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            FpInstruction::Sub(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            FpInstruction::Div(instruction) => AirConstraint::<AP>::eval(instruction, parser),
        }
    }
}

impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpInstruction<P> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            FpInstruction::Add(instruction) => Instruction::<F>::trace_layout(instruction),
            FpInstruction::Mul(instruction) => Instruction::<F>::trace_layout(instruction),
            FpInstruction::MulConst(instruction) => Instruction::<F>::trace_layout(instruction),
            FpInstruction::Inner(instruction) => Instruction::<F>::trace_layout(instruction),
            FpInstruction::Den(instruction) => Instruction::<F>::trace_layout(instruction),
            FpInstruction::Select(instruction) => Instruction::<F>::trace_layout(instruction),
            FpInstruction::Sub(instruction) => Instruction::<F>::trace_layout(instruction),
            FpInstruction::Div(instruction) => Instruction::<F>::trace_layout(instruction),
        }
    }

    fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            FpInstruction::Add(instruction) => Instruction::<F>::inputs(instruction),
            FpInstruction::Mul(instruction) => Instruction::<F>::inputs(instruction),
            FpInstruction::MulConst(instruction) => Instruction::<F>::inputs(instruction),
            FpInstruction::Inner(instruction) => Instruction::<F>::inputs(instruction),
            FpInstruction::Den(instruction) => Instruction::<F>::inputs(instruction),
            FpInstruction::Select(instruction) => Instruction::<F>::inputs(instruction),
            FpInstruction::Sub(instruction) => Instruction::<F>::inputs(instruction),
            FpInstruction::Div(instruction) => Instruction::<F>::inputs(instruction),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            FpInstruction::Add(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            FpInstruction::Mul(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            FpInstruction::MulConst(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            FpInstruction::Inner(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            FpInstruction::Den(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            FpInstruction::Select(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            FpInstruction::Sub(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            FpInstruction::Div(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
        }
    }
}

impl<P: FieldParameters> From<FpAddInstruction<P>> for FpInstruction<P> {
    fn from(instr: FpAddInstruction<P>) -> Self {
        FpInstruction::Add(instr)
    }
}

impl<P: FieldParameters> From<FpMulInstruction<P>> for FpInstruction<P> {
    fn from(instr: FpMulInstruction<P>) -> Self {
        FpInstruction::Mul(instr)
    }
}

impl<P: FieldParameters> From<FpMulConstInstruction<P>> for FpInstruction<P> {
    fn from(instr: FpMulConstInstruction<P>) -> Self {
        FpInstruction::MulConst(instr)
    }
}

impl<P: FieldParameters> From<FpInnerProductInstruction<P>> for FpInstruction<P> {
    fn from(instr: FpInnerProductInstruction<P>) -> Self {
        FpInstruction::Inner(instr)
    }
}

impl<P: FieldParameters> From<FpDenInstruction<P>> for FpInstruction<P> {
    fn from(instr: FpDenInstruction<P>) -> Self {
        FpInstruction::Den(instr)
    }
}

impl<P: FieldParameters> From<SelectInstruction<FieldRegister<P>>> for FpInstruction<P> {
    fn from(instr: SelectInstruction<FieldRegister<P>>) -> Self {
        FpInstruction::Select(instr)
    }
}

impl<P: FieldParameters> From<FpSubInstruction<P>> for FpInstruction<P> {
    fn from(instr: FpSubInstruction<P>) -> Self {
        FpInstruction::Sub(instr)
    }
}

impl<P: FieldParameters> From<FpDivInstruction<P>> for FpInstruction<P> {
    fn from(instr: FpDivInstruction<P>) -> Self {
        FpInstruction::Div(instr)
    }
}
