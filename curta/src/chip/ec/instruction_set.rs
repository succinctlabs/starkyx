use serde::{Deserialize, Serialize};

use super::scalar::LimbBitInstruction;
use super::EllipticCurve;
use crate::air::AirConstraint;
use crate::chip::field::add::FpAddInstruction;
use crate::chip::field::den::FpDenInstruction;
use crate::chip::field::div::FpDivInstruction;
use crate::chip::field::inner_product::FpInnerProductInstruction;
use crate::chip::field::instruction::{FpInstruction, FromFieldInstruction};
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::mul_const::FpMulConstInstruction;
use crate::chip::field::sub::FpSubInstruction;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::field::PrimeField64;
use crate::polynomial::parser::PolynomialParser;

pub trait ECInstructions<E: EllipticCurve>:
    FromFieldInstruction<E::BaseField> + From<LimbBitInstruction>
{
}

impl<E: EllipticCurve, T: FromFieldInstruction<E::BaseField> + From<LimbBitInstruction>>
    ECInstructions<E> for T
{
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum ECInstruction<E: EllipticCurve> {
    Fp(FpInstruction<E::BaseField>),
    LimbBit(LimbBitInstruction),
}

impl<E: EllipticCurve, AP: PolynomialParser> AirConstraint<AP> for ECInstruction<E> {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Fp(i) => i.eval(parser),
            Self::LimbBit(i) => i.eval(parser),
        }
    }
}

impl<E: EllipticCurve, F: PrimeField64> Instruction<F> for ECInstruction<E> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Fp(i) => i.write(writer, row_index),
            Self::LimbBit(i) => i.write(writer, row_index),
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        match self {
            Self::Fp(i) => i.write_to_air(writer),
            Self::LimbBit(i) => i.write_to_air(writer),
        }
    }
}

impl<E: EllipticCurve> FromFieldInstruction<E::BaseField> for ECInstruction<E> {}

impl<E: EllipticCurve> From<LimbBitInstruction> for ECInstruction<E> {
    fn from(i: LimbBitInstruction) -> Self {
        Self::LimbBit(i)
    }
}

impl<E: EllipticCurve> From<FpAddInstruction<E::BaseField>> for ECInstruction<E> {
    fn from(i: FpAddInstruction<E::BaseField>) -> Self {
        Self::Fp(i.into())
    }
}

impl<E: EllipticCurve> From<FpMulInstruction<E::BaseField>> for ECInstruction<E> {
    fn from(i: FpMulInstruction<E::BaseField>) -> Self {
        Self::Fp(i.into())
    }
}

impl<E: EllipticCurve> From<FpSubInstruction<E::BaseField>> for ECInstruction<E> {
    fn from(i: FpSubInstruction<E::BaseField>) -> Self {
        Self::Fp(i.into())
    }
}

impl<E: EllipticCurve> From<FpDivInstruction<E::BaseField>> for ECInstruction<E> {
    fn from(i: FpDivInstruction<E::BaseField>) -> Self {
        Self::Fp(i.into())
    }
}

impl<E: EllipticCurve> From<FpDenInstruction<E::BaseField>> for ECInstruction<E> {
    fn from(i: FpDenInstruction<E::BaseField>) -> Self {
        Self::Fp(i.into())
    }
}

impl<E: EllipticCurve> From<FpInnerProductInstruction<E::BaseField>> for ECInstruction<E> {
    fn from(i: FpInnerProductInstruction<E::BaseField>) -> Self {
        Self::Fp(i.into())
    }
}

impl<E: EllipticCurve> From<FpMulConstInstruction<E::BaseField>> for ECInstruction<E> {
    fn from(i: FpMulConstInstruction<E::BaseField>) -> Self {
        Self::Fp(i.into())
    }
}
