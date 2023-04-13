use std::marker::PhantomData;

use crate::arithmetic::field::FieldParameters;
use crate::arithmetic::register2::memory::MemorySlice;
use crate::arithmetic::register2::register::{Register, RegisterType};

#[derive(Debug, Clone, Copy)]
pub struct FieldRegister<P: FieldParameters> {
    register: MemorySlice,
    _marker: PhantomData<P>,
}

impl<P: FieldParameters> Register for FieldRegister<P> {
    const CELL: Option<RegisterType> = Some(RegisterType::U16);

    fn from_raw_register(register: MemorySlice) -> Self {
        Self {
            register,
            _marker: core::marker::PhantomData,
        }
    }

    fn register(&self) -> &MemorySlice {
        &self.register
    }

    fn size_of() -> usize {
        P::NB_LIMBS
    }

    fn into_raw_register(self) -> MemorySlice {
        self.register
    }
}
