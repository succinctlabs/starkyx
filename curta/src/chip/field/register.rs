use core::marker::PhantomData;

use super::parameters::FieldParameters;
use crate::air::parser::AirParser;
use crate::chip::register::cell::CellType;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::polynomial::Polynomial;

/// A register for representing a field element. The value is decomposed into a series of U16 limbs
/// which is controlled by `NB_LIMBS` in FieldParameters. Each limb is range checked using a lookup.
#[derive(Debug, Clone, Copy)]
pub struct FieldRegister<P: FieldParameters> {
    register: MemorySlice,
    _marker: PhantomData<P>,
}

impl<P: FieldParameters> RegisterSerializable for FieldRegister<P> {
    const CELL: CellType = CellType::U16;

    fn register(&self) -> &MemorySlice {
        &self.register
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self {
            register,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<P: FieldParameters> RegisterSized for FieldRegister<P> {
    fn size_of() -> usize {
        P::NB_LIMBS
    }
}

impl<P: FieldParameters> Register for FieldRegister<P> {
    type Value<T> = Polynomial<T>;

    fn eval<AP: AirParser>(&self, parser: &AP) -> Self::Value<AP::Var> {
        Polynomial::from_coefficients_slice(self.register().eval_slice(parser))
    }
}
