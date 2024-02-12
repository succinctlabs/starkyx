use core::marker::PhantomData;
use core::ops::Range;

use serde::{Deserialize, Serialize};

use super::memory::MemorySlice;
use super::{CellType, Register, RegisterSerializable};
use crate::air::parser::AirParser;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::arithmetic::expression_slice::ArithmeticExpressionSlice;
use crate::math::field::Field;

/// A helper struct for representing an array of registers. In particular, it makes it easier
/// to access the memory slice as well as converting from a memory slice to the struct.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ArrayRegister<T> {
    register: MemorySlice,
    length: usize,
    _marker: PhantomData<T>,
}

impl<T: Register> RegisterSerializable for ArrayRegister<T> {
    const CELL: CellType = T::CELL;

    fn register(&self) -> &MemorySlice {
        &self.register
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        let length = register.len() / T::size_of();
        Self {
            register,
            length,
            _marker: PhantomData,
        }
    }
}

impl<T: Register> ArrayRegister<T> {
    pub const fn uninitialized() -> Self {
        Self {
            register: MemorySlice::Global(0, 0),
            length: 0,
            _marker: PhantomData,
        }
    }

    pub const fn empty() -> Self {
        Self {
            register: MemorySlice::Global(0, 0),
            length: 0,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn from_element(element: T) -> Self {
        Self::from_register_unsafe(*element.register())
    }

    #[inline]
    pub fn get(&self, idx: usize) -> T {
        if idx >= self.len() {
            panic!(
                "Index {} out of bounds for an array of length {}",
                idx,
                self.len()
            );
        }
        self.get_unchecked(idx)
    }

    #[inline]
    pub fn first(&self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.get(0))
        }
    }

    #[inline]
    pub fn last(&self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.get(self.len() - 1))
        }
    }

    #[inline]
    pub fn get_subarray(&self, range: Range<usize>) -> Self {
        if range.end > self.len() {
            panic!(
                "End index {} out of bounds for an array of length {}",
                range.end,
                self.len()
            );
        }
        self.get_subarray_unchecked(range)
    }

    #[inline]
    fn get_unchecked(&self, idx: usize) -> T {
        let offset = T::size_of() * idx;
        match self.register {
            MemorySlice::Local(index, _) => {
                T::from_register(MemorySlice::Local(index + offset, T::size_of()))
            }
            MemorySlice::Next(index, _) => {
                T::from_register(MemorySlice::Next(index + offset, T::size_of()))
            }
            MemorySlice::Global(index, _) => {
                T::from_register(MemorySlice::Global(index + offset, T::size_of()))
            }
            MemorySlice::Public(index, _) => {
                T::from_register(MemorySlice::Public(index + offset, T::size_of()))
            }
            MemorySlice::Challenge(index, _) => {
                T::from_register(MemorySlice::Challenge(index + offset, T::size_of()))
            }
        }
    }

    #[inline]
    fn get_subarray_unchecked(&self, range: Range<usize>) -> Self {
        let offset = T::size_of() * range.start;
        let length = range.end - range.start;
        match self.register {
            MemorySlice::Local(index, _) => Self::from_register_unsafe(MemorySlice::Local(
                index + offset,
                length * T::size_of(),
            )),
            MemorySlice::Next(index, _) => {
                Self::from_register_unsafe(MemorySlice::Next(index + offset, length * T::size_of()))
            }
            MemorySlice::Global(index, _) => Self::from_register_unsafe(MemorySlice::Global(
                index + offset,
                length * T::size_of(),
            )),
            MemorySlice::Public(index, _) => Self::from_register_unsafe(MemorySlice::Public(
                index + offset,
                length * T::size_of(),
            )),
            MemorySlice::Challenge(index, _) => Self::from_register_unsafe(MemorySlice::Challenge(
                index + offset,
                length * T::size_of(),
            )),
        }
    }

    #[inline]
    pub fn iter(&self) -> ArrayIterator<T> {
        self.into_iter()
    }

    pub fn expr<F: Field>(&self) -> ArithmeticExpression<F> {
        ArithmeticExpression {
            expression: ArithmeticExpressionSlice::from_raw_register(*self.register()),
            size: self.len() * T::size_of(),
        }
    }

    #[inline]
    pub fn eval_raw_slice<'a, AP: AirParser>(&self, parser: &'a AP) -> &'a [AP::Var] {
        self.register().eval_slice(parser)
    }

    #[inline]
    pub fn eval<AP: AirParser, I: FromIterator<T::Value<AP::Var>>>(self, parser: &AP) -> I {
        self.into_iter().map(|r| r.eval(parser)).collect()
    }

    #[inline]
    pub fn eval_vec<AP: AirParser>(&self, parser: &AP) -> Vec<T::Value<AP::Var>> {
        let elem_fn = |i| self.get(i).eval(parser);
        (0..self.len()).map(elem_fn).collect()
    }

    #[inline]
    pub fn eval_array<AP: AirParser, const N: usize>(&self, parser: &AP) -> [T::Value<AP::Var>; N] {
        assert!(
            self.len() == N,
            "Array length mismatch, expected {}, got {}",
            N,
            self.len()
        );
        let elem_fn = |i| self.get(i).eval(parser);
        core::array::from_fn(elem_fn)
    }

    #[inline]
    pub fn assign_to_raw_slice<F: Copy>(&self, slice: &mut [F], value: &[T::Value<F>]) {
        let values = value
            .iter()
            .flat_map(|v| T::align(v))
            .copied()
            .collect::<Vec<_>>();
        self.register().assign_to_raw_slice(slice, &values)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArrayIterator<T: Register> {
    register: ArrayRegister<T>,
    length: usize,
    idx: usize,
}

impl<T: Register> Iterator for ArrayIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.length {
            return None;
        }
        let item = self.register.get_unchecked(self.idx);
        self.idx += 1;
        Some(item)
    }
}

impl<T: Register> IntoIterator for ArrayRegister<T> {
    type Item = T;
    type IntoIter = ArrayIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayIterator {
            register: self,
            length: self.len(),
            idx: 0,
        }
    }
}
