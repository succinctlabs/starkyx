use num::{BigUint, Num, One};
use serde::{Deserialize, Serialize};

use super::sqrt::FpSqrtInstruction;
use super::{EdwardsCurve, EdwardsParameters};
use crate::air::AirConstraint;
use crate::chip::ec::EllipticCurveParameters;
use crate::chip::field::add::FpAddInstruction;
use crate::chip::field::den::FpDenInstruction;
use crate::chip::field::div::FpDivInstruction;
use crate::chip::field::inner_product::FpInnerProductInstruction;
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::mul_const::FpMulConstInstruction;
use crate::chip::field::parameters::{FieldParameters, MAX_NB_LIMBS};
use crate::chip::field::sub::FpSubInstruction;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::TraceWriter;
use crate::math::field::PrimeField64;
use crate::polynomial::parser::PolynomialParser;

pub type Ed25519 = EdwardsCurve<Ed25519Parameters>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ed25519Parameters;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ed25519BaseField;

impl FieldParameters for Ed25519BaseField {
    const NB_BITS_PER_LIMB: usize = 16;
    const NB_LIMBS: usize = 16;
    const NB_WITNESS_LIMBS: usize = 2 * Self::NB_LIMBS - 2;
    const MODULUS: [u16; MAX_NB_LIMBS] = [
        65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535, 32767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const WITNESS_OFFSET: usize = 1usize << 20;

    fn modulus() -> BigUint {
        (BigUint::one() << 255) - BigUint::from(19u32)
    }
}

impl EllipticCurveParameters for Ed25519Parameters {
    type BaseField = Ed25519BaseField;
}

impl EdwardsParameters for Ed25519Parameters {
    const D: [u16; MAX_NB_LIMBS] = [
        30883, 4953, 19914, 30187, 55467, 16705, 2637, 112, 59544, 30585, 16505, 36039, 65139,
        11119, 27886, 20995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    fn prime_group_order() -> BigUint {
        BigUint::from(2u32).pow(252) + BigUint::from(27742317777372353535851937790883648493u128)
    }

    fn generator() -> (BigUint, BigUint) {
        let x = BigUint::from_str_radix(
            "15112221349535400772501151409588531511454012693041857206046113283949847762202",
            10,
        )
        .unwrap();
        let y = BigUint::from_str_radix(
            "46316835694926478169428394003475163141307993866256225615783033603165251855960",
            10,
        )
        .unwrap();
        (x, y)
    }
}

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
    Sqrt(FpSqrtInstruction),
}

pub trait FromEd25519FieldInstruction:
    From<FpAddInstruction<Ed25519BaseField>>
    + From<FpMulInstruction<Ed25519BaseField>>
    + From<FpSubInstruction<Ed25519BaseField>>
    + From<FpDivInstruction<Ed25519BaseField>>
    + From<FpMulConstInstruction<Ed25519BaseField>>
    + From<FpInnerProductInstruction<Ed25519BaseField>>
    + From<FpDenInstruction<Ed25519BaseField>>
    + From<FpSqrtInstruction>
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

impl From<FpSqrtInstruction> for Ed25519FpInstruction {
    fn from(instr: FpSqrtInstruction) -> Self {
        Ed25519FpInstruction::Sqrt(instr)
    }
}
