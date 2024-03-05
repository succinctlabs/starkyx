use serde::{Deserialize, Serialize};

use super::bytes::register::ByteRegister;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::memory::pointer::raw::RawPointer;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cell::CellType;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ByteArrayRegister<const N: usize>(MemorySlice);

pub type U32Register = ByteArrayRegister<4>;
pub type U64Register = ByteArrayRegister<8>;

impl<const N: usize> ByteArrayRegister<N> {
    pub fn to_le_bytes(&self) -> ArrayRegister<ByteRegister> {
        ArrayRegister::from_register_unsafe(self.0)
    }

    pub fn to_le_limbs<const M: usize>(&self) -> ArrayRegister<ByteArrayRegister<M>> {
        assert!(N % M == 0);
        ArrayRegister::from_register_unsafe(self.0)
    }

    pub fn from_limbs<const M: usize>(register: &ArrayRegister<ByteArrayRegister<M>>) -> Self {
        assert!(N % M == 0);
        Self::from_register_unsafe(*register.register())
    }
}

impl<const N: usize> RegisterSerializable for ByteArrayRegister<N> {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self(register)
    }
}

impl<const N: usize> RegisterSized for ByteArrayRegister<N> {
    fn size_of() -> usize {
        N
    }
}

impl<const N: usize> Register for ByteArrayRegister<N> {
    type Value<T> = [T; N];

    fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T> {
        let elem_fn = |i| slice[i];
        core::array::from_fn(elem_fn)
    }

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        value
    }
}

impl MemoryValue for U32Register {
    fn num_challenges() -> usize {
        0
    }

    fn compress<L: crate::chip::AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        ptr: RawPointer,
        time: &Time<L::Field>,
        _: &ArrayRegister<CubicRegister>,
    ) -> CubicRegister {
        let bytes = self.to_le_bytes();
        let mut acc = ArithmeticExpression::zero();

        for (i, byte) in bytes.iter().enumerate() {
            let two_i = ArithmeticExpression::from(L::Field::from_canonical_u32(1 << (8 * i)));
            acc = acc + two_i * byte.expr();
        }

        let zero = ArithmeticExpression::zero();
        let acc_expression = CubicElement([acc, time.expr(), zero]);

        ptr.accumulate_cubic(builder, acc_expression)
    }
}

impl MemoryValue for U64Register {
    fn num_challenges() -> usize {
        0
    }

    fn compress<L: crate::chip::AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        ptr: RawPointer,
        time: &Time<L::Field>,
        _: &ArrayRegister<CubicRegister>,
    ) -> CubicRegister {
        let bytes = self.to_le_bytes();
        let low_bytes = bytes.get_subarray(0..4);
        let high_bytes = bytes.get_subarray(4..8);

        let mut acc_low = ArithmeticExpression::zero();
        let mut acc_high = ArithmeticExpression::zero();

        for (i, (byte_low, byte_high)) in low_bytes.iter().zip(high_bytes).enumerate() {
            let two_i = ArithmeticExpression::from(L::Field::from_canonical_u32(1 << (8 * i)));
            acc_low = acc_low + two_i.clone() * byte_low.expr();
            acc_high = acc_high + two_i * byte_high.expr();
        }
        let accumulated_expr = CubicElement([acc_low, acc_high, time.expr()]);
        ptr.accumulate_cubic(builder, accumulated_expr)
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::AirParameters;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct RegisterConversionTest;

    impl AirParameters for RegisterConversionTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 2;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
    }

    #[test]
    fn test_byte_array_register() {
        type L = RegisterConversionTest;

        let mut builder = AirBuilder::<L>::new();

        const N: usize = 8;
        const M: usize = 4;

        let a = builder.alloc::<ByteArrayRegister<N>>();

        let a_as_limbs = a.to_le_limbs::<M>();

        let b = ByteArrayRegister::<N>::from_limbs(&a_as_limbs);

        builder.assert_equal(&a, &b);
    }
}
