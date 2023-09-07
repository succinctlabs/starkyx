use serde::{Deserialize, Serialize};

use super::bytes::register::ByteRegister;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cell::CellType;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ByteArrayRegister<const N: usize>(MemorySlice);

pub type U8Register = ByteArrayRegister<1>;
pub type U16Register = ByteArrayRegister<2>;
pub type U32Register = ByteArrayRegister<4>;
pub type U64Register = ByteArrayRegister<8>;

impl<const N: usize> ByteArrayRegister<N> {
    pub fn to_le_bytes(&self) -> ArrayRegister<ByteRegister> {
        ArrayRegister::from_register_unsafe(self.0)
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

/// N is the number of bytes in the input register.
/// M is the number of bytes that the input register is split into.
pub fn to_le_limbs<const N: usize, const M: usize>(
    register: &ByteArrayRegister<N>,
) -> ArrayRegister<ByteArrayRegister<M>> {
    assert!(N % M == 0);
    ArrayRegister::from_register_unsafe(register.0)
}

pub fn from_limbs<const N: usize, const M: usize>(
    register: &ArrayRegister<ByteArrayRegister<M>>,
) -> ByteArrayRegister<N> {
    assert!(N % M == 0);

    ByteArrayRegister::<N>::from_register_unsafe(*register.register())
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::AirParameters;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct RegisterConversionTest;

    impl AirParameters for RegisterConversionTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 2;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_byte_array_register() {
        type L = RegisterConversionTest;

        let mut builder = AirBuilder::<L>::new();

        const N: usize = 8;
        const M: usize = 4;

        let a = builder.alloc::<ByteArrayRegister<N>>();

        let a_as_limbs = to_le_limbs::<N, M>(&a);

        let b = from_limbs::<N, M>(&a_as_limbs);

        builder.assert_equal(&a, &b);
    }
}
