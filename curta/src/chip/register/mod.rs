use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use self::cell::CellType;
use self::memory::MemorySlice;
use super::arithmetic::expression::ArithmeticExpression;
use super::arithmetic::expression_slice::ArithmeticExpressionSlice;
use crate::air::parser::AirParser;
use crate::math::prelude::*;

pub mod array;
pub mod bit;
pub mod cell;
pub mod cubic;
pub mod element;
pub mod memory;
pub mod slice;
pub mod u16;

/// Adds serialization and deserialization to a register for converting between the canonical type
/// and `MemorySlice`.
pub trait RegisterSerializable
where
    Self: Sized,
{
    const CELL: CellType;

    fn register(&self) -> &MemorySlice;

    /// Initializes the register given a memory slice without checks on length. Use
    /// `from_register` if you want to check the length.
    fn from_register_unsafe(register: MemorySlice) -> Self;

    /// Returns the register but in the next row.
    fn next(&self) -> Self {
        Self::from_register_unsafe(self.register().next())
    }

    /// Returns `true` if the register is a trace register.
    fn is_trace(&self) -> bool {
        self.register().is_trace()
    }
}

/// Ensures that the register has a fixed size.
pub trait RegisterSized {
    /// Returns the expected size of the register in cells.
    fn size_of() -> usize;
}

/// A register is a slice of memory in the trace that is supposed to represent a specific type of
/// data. A register can be thought as a compiler provided type--it should not be necessary to
/// ever use the register type directly to access values. If you want to access the values, you
/// should instead compose multiple different register types into a struct.
pub trait Register:
    RegisterSerializable
    + RegisterSized
    + 'static
    + Debug
    + Sized
    + Clone
    + Send
    + Sync
    + Copy
    + Serialize
    + for<'de> Deserialize<'de>
{
    type Value<T>;

    // Determins the layout of Value<T> as a slice that can be assigned to the trace.
    fn align<T>(value: &Self::Value<T>) -> &[T];

    // Gets a new value from a slice.
    fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T>;

    fn eval<AP: AirParser>(&self, parser: &AP) -> Self::Value<AP::Var> {
        Self::value_from_slice(self.register().eval_slice(parser))
    }

    /// Initializes the register given a memory slice with checks on length.
    fn from_register(register: MemorySlice) -> Self {
        if register.len() != Self::size_of() {
            panic!("Invalid register length");
        }
        Self::from_register_unsafe(register)
    }

    fn assign_to_raw_slice<T: Copy>(&self, slice: &mut [T], value: &Self::Value<T>) {
        self.register()
            .assign_to_raw_slice(slice, Self::align(value))
    }

    fn read_from_slice<T: Copy>(&self, slice: &[T]) -> Self::Value<T> {
        Self::value_from_slice(self.register().read_from_slice(slice))
    }

    fn expr<F: Field>(&self) -> ArithmeticExpression<F> {
        ArithmeticExpression {
            expression: ArithmeticExpressionSlice::from_raw_register(*self.register()),
            size: Self::size_of(),
        }
    }
}

impl RegisterSerializable for MemorySlice {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        self
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        register
    }
}
