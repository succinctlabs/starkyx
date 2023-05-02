use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::{CellType, Register, RegisterSerializable};
use crate::curta::constraint::expression::ArithmeticExpression;
use crate::curta::register::memory::MemorySlice;

/// A helper struct for representing an array of registers. In particular, it makes it easier
/// to access the memory slice as well as converting from a memory slice to the struct.
#[derive(Debug, Clone, Copy)]
pub struct ArrayRegister<T: Register> {
    register: MemorySlice,
    length: usize,
    _marker: PhantomData<T>,
}

impl<T: Register> RegisterSerializable for ArrayRegister<T> {
    const CELL: Option<CellType> = T::CELL;

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

    pub fn get(&self, idx: usize) -> T {
        if idx >= self.len() {
            panic!("Index out of bounds");
        }
        let offset = T::size_of() * idx;
        match self.register {
            MemorySlice::Local(col, _) => {
                T::from_register(MemorySlice::Local(col + offset, T::size_of()))
            }
            MemorySlice::Next(col, _) => {
                T::from_register(MemorySlice::Next(col + offset, T::size_of()))
            }
            MemorySlice::First(col, _) => {
                T::from_register(MemorySlice::First(col + offset, T::size_of()))
            }
            MemorySlice::Last(col, _) => {
                T::from_register(MemorySlice::Last(col + offset, T::size_of()))
            }
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(T::size_of() == 1, "Only works on registers of size 1");
        let sliced_register = match self.register {
            MemorySlice::Local(col, size) => MemorySlice::Local(col + start, size - start),
            MemorySlice::Next(col, size) => MemorySlice::Next(col + start, size - start),
            MemorySlice::First(col, size) => MemorySlice::First(col + start, size - start),
            MemorySlice::Last(col, size) => MemorySlice::Last(col + start, size - start),
        };
        Self {
            register: sliced_register,
            length: end - start,
            _marker: self._marker,
        }
    }

    pub fn expr<F: RichField + Extendable<D>, const D: usize>(&self) -> ArithmeticExpression<F, D> {
        self.register.expr()
    }

    pub fn expr_sum<F: RichField + Extendable<D>, const D: usize>(
        &self,
    ) -> ArithmeticExpression<F, D> {
        assert!(T::size_of() == 1, "Only works on registers of size 1");
        match self.register {
            MemorySlice::Local(col, size) => {
                let mut sum = MemorySlice::Local(col, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::Local(col + i, 1).expr();
                }
                sum
            }
            MemorySlice::Next(col, size) => {
                let mut sum = MemorySlice::Next(col, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::Next(col + i, 1).expr();
                }
                sum
            }
            MemorySlice::First(col, size) => {
                let mut sum = MemorySlice::First(col, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::First(col + i, 1).expr();
                }
                sum
            }
            MemorySlice::Last(col, size) => {
                let mut sum = MemorySlice::Last(col, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::Last(col + i, 1).expr();
                }
                sum
            }
        }
    }

    pub fn expr_be_u32<F: RichField + Extendable<D>, const D: usize>(
        &self,
    ) -> ArithmeticExpression<F, D> {
        assert!(self.len() == 32, "Only works on arrays of size 32");
        assert!(T::size_of() == 1, "Only works on registers of size 1");
        match self.register {
            MemorySlice::Local(col, size) => {
                let mut digit = F::TWO;
                let mut sum = MemorySlice::Local(col + 31, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::Local(col + 31 - i, 1).expr() * digit;
                    digit *= F::TWO;
                }
                sum
            }
            MemorySlice::Next(col, size) => {
                let mut digit = F::TWO;
                let mut sum = MemorySlice::Next(col + 31, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::Next(col + 31 - i, 1).expr() * digit;
                    digit *= F::TWO;
                }
                sum
            }
            MemorySlice::First(col, size) => {
                let mut digit = F::TWO;
                let mut sum = MemorySlice::First(col + 31, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::First(col + 31 - i, 1).expr() * digit;
                    digit *= F::TWO;
                }
                sum
            }
            MemorySlice::Last(col, size) => {
                let mut digit = F::TWO;
                let mut sum = MemorySlice::Last(col + 31, 1).expr();
                for i in 1..size {
                    sum = sum + MemorySlice::Last(col + 31 - i, 1).expr() * digit;
                    digit *= F::TWO;
                }
                sum
            }
        }
    }
}
