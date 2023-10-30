use core::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::air::parser::AirParser;
use crate::trace::view::{TraceView, TraceViewMut};

/// A contiguous chunk of memory in the trace and Stark data.
/// Corresponds to a slice in vars.local_values, vars.next_values, vars.public_inputs,
/// or vars.challenges.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Serialize, Deserialize)]
pub enum MemorySlice {
    /// A slice of the current row.
    Local(usize, usize),
    /// A slice of the next row.
    Next(usize, usize),
    /// A slice of public inputs
    Public(usize, usize),
    /// A slice of values from global variables of the air
    Global(usize, usize),
    /// A slice of values coming from verifier challenges
    Challenge(usize, usize),
}

impl MemorySlice {
    #[inline]
    pub fn is_next(&self) -> bool {
        matches!(self, MemorySlice::Next(_, _))
    }

    #[inline]
    pub fn is_trace(&self) -> bool {
        matches!(self, MemorySlice::Local(_, _) | MemorySlice::Next(_, _))
    }

    #[inline]
    pub fn next(&self) -> Self {
        match self {
            MemorySlice::Local(index, length) => MemorySlice::Next(*index, *length),
            _ => panic!("Invalid register type for the next register"),
        }
    }

    #[inline]
    pub fn get_range(&self) -> (usize, usize) {
        match self {
            MemorySlice::Local(index, length) => (*index, *index + length),
            MemorySlice::Next(index, length) => (*index, *index + length),
            MemorySlice::Global(index, length) => (*index, *index + length),
            MemorySlice::Public(index, length) => (*index, *index + length),
            MemorySlice::Challenge(index, length) => (*index, *index + length),
        }
    }

    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            MemorySlice::Local(index, _) => *index,
            MemorySlice::Next(index, _) => *index,
            MemorySlice::Global(index, _) => *index,
            MemorySlice::Public(index, _) => *index,
            MemorySlice::Challenge(index, _) => *index,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        match self {
            MemorySlice::Local(_, length) => *length,
            MemorySlice::Next(_, length) => *length,
            MemorySlice::Global(_, length) => *length,
            MemorySlice::Public(_, length) => *length,
            MemorySlice::Challenge(_, length) => *length,
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn eval_slice<'a, AP: AirParser>(&self, parser: &'a AP) -> &'a [AP::Var] {
        match self {
            MemorySlice::Local(index, length) => &parser.local_slice()[*index..*index + length],
            MemorySlice::Next(index, length) => &parser.next_slice()[*index..*index + length],
            MemorySlice::Global(index, length) => &parser.global_slice()[*index..*index + length],
            MemorySlice::Public(index, length) => &parser.public_slice()[*index..*index + length],
            MemorySlice::Challenge(index, length) => {
                &parser.challenge_slice()[*index..*index + length]
            }
        }
    }

    #[inline]
    pub fn read<'a, T: Copy>(&self, trace_view: &'a TraceView<T>, row_index: usize) -> &'a [T] {
        match self {
            MemorySlice::Local(index, length) => {
                &trace_view.row(row_index)[*index..*index + length]
            }
            MemorySlice::Next(index, length) => {
                &trace_view.row(row_index + 1)[*index..*index + length]
            }
            MemorySlice::Global(_, _) => {
                unreachable!("Cannot read from global inputs with this method")
            }
            MemorySlice::Public(_, _) => {
                unreachable!("Cannot read from public inputs with this method")
            }
            MemorySlice::Challenge(_, _) => {
                unreachable!("Cannot read from challenges with this method")
            }
        }
    }

    #[inline]
    pub fn read_from_slice<'a, T: Copy>(&self, slice: &'a [T]) -> &'a [T] {
        match self {
            MemorySlice::Local(index, length) => &slice[*index..*index + length],
            MemorySlice::Next(_, _) => {
                unreachable!("Cannot read from next row with this method")
            }
            MemorySlice::Global(index, length) => &slice[*index..*index + length],
            MemorySlice::Public(index, length) => &slice[*index..*index + length],
            MemorySlice::Challenge(index, length) => &slice[*index..*index + length],
        }
    }

    /// Assigns a value to the location specified by the memory slice
    ///
    /// The values are read from `value` starting at `local_index`. The new local index is returned.
    #[inline]
    pub fn assign<T: Copy>(
        &self,
        trace_view: &mut TraceViewMut<T>,
        local_index: usize,
        value: &[T],
        row_index: usize,
    ) -> usize {
        let value = &value[local_index..local_index + self.len()];
        match self {
            MemorySlice::Local(index, length) => {
                trace_view.row_mut(row_index)[*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Next(index, length) => {
                trace_view.row_mut(row_index + 1)[*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Global(_, _) => {
                unreachable!("Cannot assign to global inputs with this method")
            }
            MemorySlice::Public(_, _) => {
                unreachable!("Cannot assign to public inputs with this method")
            }
            MemorySlice::Challenge(_, _) => unreachable!("Cannot assign to challenges"),
        }
        local_index + self.len()
    }

    #[inline]
    pub fn assign_to_raw_slice<T: Copy>(&self, row: &mut [T], value: &[T]) {
        match self {
            MemorySlice::Local(index, length) => {
                row[*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Next(_, _) => unreachable!("Cannot assign to next row with this method"),
            MemorySlice::Global(index, length) => {
                row[*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Public(index, length) => {
                row[*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Challenge(_, _) => unreachable!("Cannot assign to challenges"),
        }
    }
}

impl Hash for MemorySlice {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.get_range().hash(state);
        match self {
            MemorySlice::Local(_, _) => "local".hash(state),
            MemorySlice::Next(_, _) => "next".hash(state),
            MemorySlice::Global(_, _) => "public".hash(state),
            MemorySlice::Public(_, _) => "public".hash(state),
            MemorySlice::Challenge(_, _) => "challenge".hash(state),
        }
    }
}
