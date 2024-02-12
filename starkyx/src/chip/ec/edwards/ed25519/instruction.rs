use serde::{Deserialize, Serialize};

use super::params::{Ed25519, Ed25519BaseField};
use super::sqrt::Ed25519FpSqrtInstruction;
use crate::air::AirConstraint;
use crate::chip::ec::scalar::LimbBitInstruction;
use crate::chip::ec::ECInstruction;
use crate::chip::field::add::FpAddInstruction;
use crate::chip::field::den::FpDenInstruction;
use crate::chip::field::div::FpDivInstruction;
use crate::chip::field::inner_product::FpInnerProductInstruction;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::mul_const::FpMulConstInstruction;
use crate::chip::field::sub::FpSubInstruction;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::field::PrimeField64;
use crate::polynomial::parser::PolynomialParser;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum Ed25519FpInstruction {
    EC(ECInstruction<Ed25519>),
    Sqrt(Ed25519FpSqrtInstruction),
}

impl FromFieldInstruction<Ed25519BaseField> for Ed25519FpInstruction {}

impl From<Ed25519FpSqrtInstruction> for Ed25519FpInstruction {
    fn from(i: Ed25519FpSqrtInstruction) -> Self {
        Self::Sqrt(i)
    }
}

impl<AP: PolynomialParser> AirConstraint<AP> for Ed25519FpInstruction {
    fn eval(&self, parser: &mut AP) {
        match self {
            Ed25519FpInstruction::EC(instruction) => AirConstraint::<AP>::eval(instruction, parser),
            Ed25519FpInstruction::Sqrt(instruction) => {
                AirConstraint::<AP>::eval(instruction, parser)
            }
        }
    }
}

impl<F: PrimeField64> Instruction<F> for Ed25519FpInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Ed25519FpInstruction::EC(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
            Ed25519FpInstruction::Sqrt(instruction) => {
                Instruction::<F>::write(instruction, writer, row_index)
            }
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        match self {
            Ed25519FpInstruction::EC(instruction) => {
                Instruction::<F>::write_to_air(instruction, writer)
            }
            Ed25519FpInstruction::Sqrt(instruction) => {
                Instruction::<F>::write_to_air(instruction, writer)
            }
        }
    }
}

impl From<LimbBitInstruction> for Ed25519FpInstruction {
    fn from(i: LimbBitInstruction) -> Self {
        Self::EC(i.into())
    }
}

impl From<FpAddInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(i: FpAddInstruction<Ed25519BaseField>) -> Self {
        Self::EC(i.into())
    }
}

impl From<FpMulInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(i: FpMulInstruction<Ed25519BaseField>) -> Self {
        Self::EC(i.into())
    }
}

impl From<FpSubInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(i: FpSubInstruction<Ed25519BaseField>) -> Self {
        Self::EC(i.into())
    }
}

impl From<FpDivInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(i: FpDivInstruction<Ed25519BaseField>) -> Self {
        Self::EC(i.into())
    }
}

impl From<FpDenInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(i: FpDenInstruction<Ed25519BaseField>) -> Self {
        Self::EC(i.into())
    }
}

impl From<FpInnerProductInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(i: FpInnerProductInstruction<Ed25519BaseField>) -> Self {
        Self::EC(i.into())
    }
}

impl From<FpMulConstInstruction<Ed25519BaseField>> for Ed25519FpInstruction {
    fn from(i: FpMulConstInstruction<Ed25519BaseField>) -> Self {
        Self::EC(i.into())
    }
}
