use core::marker::PhantomData;

use super::memory::MemorySlice;
use super::{CellType, Register, RegisterSerializable};
use crate::air::parser::AirParser;

/// A helper struct for representing an array of registers. In particular, it makes it easier
/// to access the memory slice as well as converting from a memory slice to the struct.
#[derive(Debug, Clone, Copy)]
pub struct ArrayRegister<T: Register> {
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
            panic!("Index out of bounds");
        }
        self.get_unchecked(idx)
    }

    #[inline]
    fn get_unchecked(&self, idx: usize) -> T {
        let offset = T::size_of() * idx;
        match self.register {
            MemorySlice::Local(col, _) => {
                T::from_register(MemorySlice::Local(col + offset, T::size_of()))
            }
            MemorySlice::Next(col, _) => {
                T::from_register(MemorySlice::Next(col + offset, T::size_of()))
            }
            MemorySlice::Public(col, _) => {
                T::from_register(MemorySlice::Public(col + offset, T::size_of()))
            }
            MemorySlice::Challenge(col, _) => {
                T::from_register(MemorySlice::Challenge(col + offset, T::size_of()))
            }
        }
    }

    pub fn iter(&self) -> ArrayIterator<T> {
        self.into_iter()
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
    pub fn eval_vec<AP: AirParser>(&self, parser: &AP) -> Vec<AP::Var> {
        self.register().eval_slice(parser).to_vec()
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
