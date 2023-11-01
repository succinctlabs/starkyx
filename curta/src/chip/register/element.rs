use serde::{Deserialize, Serialize};

use super::array::ArrayRegister;
use super::cell::CellType;
use super::cubic::CubicRegister;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};
use crate::chip::builder::AirBuilder;
use crate::chip::memory::pointer::raw::RawPointer;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::machine::builder::ops::{Add, Mul};
use crate::machine::builder::Builder;
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::*;

/// A register for a single element/column in the trace. The value is not constrainted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ElementRegister(MemorySlice);

impl RegisterSerializable for ElementRegister {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        ElementRegister(register)
    }
}

impl RegisterSized for ElementRegister {
    fn size_of() -> usize {
        1
    }
}

impl Register for ElementRegister {
    type Value<T> = T;

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        std::slice::from_ref(value)
    }

    fn value_from_slice<T: Clone>(slice: &[T]) -> Self::Value<T> {
        slice[0].clone()
    }
}

impl MemoryValue for ElementRegister {
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
        let value = CubicElement([time.expr(), self.expr(), L::Field::ZERO.into()]);
        ptr.accumulate_cubic(builder, value)
    }
}

impl<B: Builder> Add<B> for ElementRegister {
    type Output = Self;

    fn add(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.expression(self.expr() + rhs.expr())
    }
}

impl<B: Builder> Mul<B> for ElementRegister {
    type Output = Self;

    fn mul(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.expression(self.expr() * rhs.expr())
    }
}
