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

    #[inline]
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
}
