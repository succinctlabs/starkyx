use super::builder::BytesBuilder;
use crate::chip::register::bit::BitRegister;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::{ByteArrayRegister, U32Register, U64Register};
use crate::chip::AirParameters;
use crate::machine::builder::ops::{Adc, Add, And, Not, RotateRight, Shr, Xor};
use crate::machine::builder::Builder;

impl<L: AirParameters, const N: usize> And<BytesBuilder<L>> for &ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn and(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.api.bitwise_and(self, rhs, &mut builder.operations)
    }
}

impl<L: AirParameters, const N: usize> And<BytesBuilder<L>> for ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn and(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.and(&self, &rhs)
    }
}

impl<L: AirParameters, const N: usize> Not<BytesBuilder<L>> for &ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn not(self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.api.bitwise_not(self, &mut builder.operations)
    }
}

impl<L: AirParameters, const N: usize> Not<BytesBuilder<L>> for ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn not(self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.not(&self)
    }
}

impl<L: AirParameters, const N: usize> Xor<BytesBuilder<L>> for &ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn xor(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.api.bitwise_xor(self, rhs, &mut builder.operations)
    }
}

impl<L: AirParameters, const N: usize> Xor<BytesBuilder<L>> for ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn xor(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.xor(&self, &rhs)
    }
}

impl<L: AirParameters, const N: usize> Shr<BytesBuilder<L>, usize> for &ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn shr(self, rhs: usize, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.api.bit_shr(self, rhs, &mut builder.operations)
    }
}

impl<L: AirParameters, const N: usize> Shr<BytesBuilder<L>, usize> for ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn shr(self, rhs: usize, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.shr(&self, rhs)
    }
}

impl<L: AirParameters, const N: usize> RotateRight<BytesBuilder<L>, usize> for &ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn rotate_right(self, rhs: usize, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder
            .api
            .bit_rotate_right(self, rhs, &mut builder.operations)
    }
}

impl<L: AirParameters, const N: usize> RotateRight<BytesBuilder<L>, usize> for ByteArrayRegister<N>
where
    L::Instruction: UintInstructions,
{
    type Output = ByteArrayRegister<N>;

    fn rotate_right(self, rhs: usize, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.rotate_right(&self, rhs)
    }
}

impl<L: AirParameters> Adc<BytesBuilder<L>> for &U32Register
where
    L::Instruction: UintInstructions,
{
    type Output = (U32Register, BitRegister);

    fn adc(self, rhs: Self, carry: BitRegister, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder
            .api
            .carrying_add_u32(self, rhs, &Some(carry), &mut builder.operations)
    }
}

impl<L: AirParameters> Adc<BytesBuilder<L>> for U32Register
where
    L::Instruction: UintInstructions,
{
    type Output = (U32Register, BitRegister);

    fn adc(self, rhs: Self, carry: BitRegister, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.carrying_add(&self, &rhs, carry)
    }
}

impl<L: AirParameters> Add<BytesBuilder<L>> for &U32Register
where
    L::Instruction: UintInstructions,
{
    type Output = U32Register;

    fn add(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.api.add_u32(self, rhs, &mut builder.operations)
    }
}

impl<L: AirParameters> Add<BytesBuilder<L>> for U32Register
where
    L::Instruction: UintInstructions,
{
    type Output = U32Register;

    fn add(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.add(&self, &rhs)
    }
}

impl<L: AirParameters> Adc<BytesBuilder<L>> for &U64Register
where
    L::Instruction: UintInstructions,
{
    type Output = (U64Register, BitRegister);

    fn adc(self, rhs: Self, carry: BitRegister, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder
            .api
            .carrying_add_u64(self, rhs, &Some(carry), &mut builder.operations)
    }
}

impl<L: AirParameters> Adc<BytesBuilder<L>> for U64Register
where
    L::Instruction: UintInstructions,
{
    type Output = (U64Register, BitRegister);

    fn adc(self, rhs: Self, carry: BitRegister, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.carrying_add(&self, &rhs, carry)
    }
}

impl<L: AirParameters> Add<BytesBuilder<L>> for &U64Register
where
    L::Instruction: UintInstructions,
{
    type Output = U64Register;

    fn add(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.api.add_u64(self, rhs, &mut builder.operations)
    }
}

impl<L: AirParameters> Add<BytesBuilder<L>> for U64Register
where
    L::Instruction: UintInstructions,
{
    type Output = U64Register;

    fn add(self, rhs: Self, builder: &mut BytesBuilder<L>) -> Self::Output {
        builder.add(&self, &rhs)
    }
}
