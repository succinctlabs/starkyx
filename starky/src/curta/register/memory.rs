use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::curta::air::parser::AirParser;
use crate::curta::constraint::arithmetic::{ArithmeticExpression, ArithmeticExpressionSlice};

/// A contiguous chunk of memory in the trace and Stark data.
/// Corresponds to a slice in vars.local_values, vars.next_values, vars.public_inputs,
/// or vars.challenges.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum MemorySlice {
    /// A slice of the current row.
    Local(usize, usize),
    /// A slice of the next row.
    Next(usize, usize),
    /// A slice of public inputs
    Public(usize, usize),
    /// A slice of values coming from verifier challenges
    Challenge(usize, usize),
}

impl MemorySlice {
    #[inline]
    pub fn is_next(&self) -> bool {
        matches!(self, MemorySlice::Next(_, _))
    }

    pub fn next(&self) -> Self {
        match self {
            MemorySlice::Local(index, length) => MemorySlice::Next(*index, *length),
            _ => panic!("Invalid register type for the next register"),
        }
    }

    #[inline]
    pub const fn get_range(&self) -> (usize, usize) {
        match self {
            MemorySlice::Local(index, length) => (*index, *index + length),
            MemorySlice::Next(index, length) => (*index, *index + length),
            MemorySlice::Public(index, length) => (*index, *index + length),
            MemorySlice::Challenge(index, length) => (*index, *index + length),
        }
    }

    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            MemorySlice::Local(index, _) => *index,
            MemorySlice::Next(index, _) => *index,
            MemorySlice::Public(index, _) => *index,
            MemorySlice::Challenge(index, _) => *index,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        match self {
            MemorySlice::Local(_, length) => *length,
            MemorySlice::Next(_, length) => *length,
            MemorySlice::Public(_, length) => *length,
            MemorySlice::Challenge(_, length) => *length,
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn read<T: Copy>(&self, trace_rows: &[Vec<T>], value: &mut [T], row_index: usize) {
        match self {
            MemorySlice::Local(index, length) => {
                value.copy_from_slice(&trace_rows[row_index][*index..*index + length]);
            }
            MemorySlice::Next(index, length) => {
                value.copy_from_slice(&trace_rows[row_index + 1][*index..*index + length]);
            }
            _ => panic!("Cannot read from a non-local register"),
        }
    }

    #[inline]
    pub fn assign<T: Copy>(
        &self,
        trace_rows: &mut [Vec<T>],
        local_index: usize,
        value: &[T],
        row_index: usize,
    ) -> usize {
        let value = &value[local_index..local_index + self.len()];
        match self {
            MemorySlice::Local(index, length) => {
                trace_rows[row_index][*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Next(index, length) => {
                trace_rows[row_index + 1][*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Public(_, _) => unimplemented!("Cannot assign to public inputs"),
            MemorySlice::Challenge(_, _) => unimplemented!("Cannot assign to challenges"),
        }
        local_index + self.len()
    }

    pub fn expr<F: RichField + Extendable<D>, const D: usize>(&self) -> ArithmeticExpression<F, D> {
        ArithmeticExpression {
            expression: ArithmeticExpressionSlice::from_raw_register(*self),
            size: self.len(),
        }
    }

    #[inline]
    pub fn eval_slice<'a, AP: AirParser>(&self, parser: &'a AP) -> &'a [AP::Var] {
        match self {
            MemorySlice::Local(index, length) => &parser.local_slice()[*index..*index + length],
            MemorySlice::Next(index, length) => &parser.next_slice()[*index..*index + length],
            MemorySlice::Public(index, length) => &parser.public_slice()[*index..*index + length],
            MemorySlice::Challenge(index, length) => {
                &parser.challenge_slice()[*index..*index + length]
            }
        }
    }
}
