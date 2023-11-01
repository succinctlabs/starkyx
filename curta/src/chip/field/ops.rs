use super::instruction::FromFieldInstruction;
use super::parameters::FieldParameters;
use super::register::FieldRegister;
use crate::machine::builder::ops::{Add, Div, Mul, One, Sub, Zero};
use crate::machine::builder::Builder;

impl<B: Builder, P: FieldParameters> Add<B> for FieldRegister<P>
where
    B::Instruction: FromFieldInstruction<P>,
{
    type Output = Self;

    fn add(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.api().fp_add(&self, &rhs)
    }
}

impl<B: Builder, P: FieldParameters> Sub<B> for FieldRegister<P>
where
    B::Instruction: FromFieldInstruction<P>,
{
    type Output = Self;

    fn sub(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.api().fp_sub(&self, &rhs)
    }
}

impl<B: Builder, P: FieldParameters> Mul<B> for FieldRegister<P>
where
    B::Instruction: FromFieldInstruction<P>,
{
    type Output = Self;

    fn mul(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.api().fp_mul(&self, &rhs)
    }
}

impl<B: Builder, P: FieldParameters> Div<B> for FieldRegister<P>
where
    B::Instruction: FromFieldInstruction<P>,
{
    type Output = Self;

    fn div(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.api().fp_div(&self, &rhs)
    }
}

impl<B: Builder, P: FieldParameters> Zero<B> for FieldRegister<P>
where
    B::Instruction: FromFieldInstruction<P>,
{
    type Output = Self;

    fn zero(builder: &mut B) -> Self::Output {
        builder.api().fp_zero()
    }
}

impl<B: Builder, P: FieldParameters> One<B> for FieldRegister<P>
where
    B::Instruction: FromFieldInstruction<P>,
{
    type Output = Self;

    fn one(builder: &mut B) -> Self::Output {
        builder.api().fp_one()
    }
}
