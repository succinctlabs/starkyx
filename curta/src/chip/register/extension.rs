use super::cell::CellType;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::plonky2::field::cubic::element::CubicElement;

/// A register for a single element/column in the trace. The value is not constrainted.
#[derive(Debug, Clone, Copy)]
pub struct ExtensionRegister<const D: usize>(MemorySlice);

impl<const D: usize> RegisterSerializable for ExtensionRegister<D> {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        ExtensionRegister(register)
    }
}

impl<const D: usize> RegisterSized for ExtensionRegister<D> {
    fn size_of() -> usize {
        D
    }
}

impl<const D: usize> Register for ExtensionRegister<D> {
    type Value<T> = [T; D];

    fn eval<AP: AirParser>(&self, parser: &AP) -> Self::Value<AP::Var> {
        let slice = self.register().eval_slice(parser);
        debug_assert!(
            slice.len() == D,
            "Slice length mismatch for register {:?} (expected {}, got {})",
            self,
            D,
            slice.len()
        );
        core::array::from_fn(|i| slice[i])
    }

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        value
    }
}

impl ExtensionRegister<3> {
    #[inline]
    pub fn eval_extension<E, AP: CubicParser<E>>(&self, parser: &AP) -> CubicElement<AP::Var>
    where
        E: CubicParameters<AP::Field>,
    {
        parser.element_from_base_slice(&self.eval(parser))
    }
}
