use super::bytes::register::ByteRegister;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cell::CellType;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};

#[derive(Debug, Clone, Copy)]
pub struct ByteArrayRegister<const N: usize>(MemorySlice);

pub type U8Register = ByteArrayRegister<1>;
pub type U16Register = ByteArrayRegister<2>;
pub type U32Register = ByteArrayRegister<4>;
pub type U64Register = ByteArrayRegister<8>;
pub type U128Register = ByteArrayRegister<16>;

impl<const N: usize> ByteArrayRegister<N> {
    pub fn bytes(&self) -> ArrayRegister<ByteRegister> {
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
