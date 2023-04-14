use plonky2::field::extension::FieldExtension;
use plonky2::field::packed::PackedField;
use plonky2::iop::ext_target::ExtensionTarget;

use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

/// A row-wise contiguous chunk of memory in the trace. Corresponds to a slice in vars.local_values
/// or vars.next_values.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum MemorySlice {
    Local(usize, usize),
    Next(usize, usize),

    // Not sure if these are needed
    First(usize, usize),
    Last(usize, usize),
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
            MemorySlice::First(index, length) => (*index, *index + length),
            MemorySlice::Last(index, length) => (*index, *index + length),
        }
    }

    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            MemorySlice::Local(index, _) => *index,
            MemorySlice::Next(index, _) => *index,
            MemorySlice::First(index, _) => *index,
            MemorySlice::Last(index, _) => *index,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        match self {
            MemorySlice::Local(_, length) => *length,
            MemorySlice::Next(_, length) => *length,
            MemorySlice::First(_, length) => *length,
            MemorySlice::Last(_, length) => *length,
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn read<T: Copy>(&self, trace_rows: &mut [Vec<T>], value: &mut [T], row_index: usize) {
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
        value: &mut [T],
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
            MemorySlice::First(index, length) => {
                trace_rows[0][*index..*index + length].copy_from_slice(value);
            }
            MemorySlice::Last(index, length) => {
                trace_rows[trace_rows.len() - 1][*index..*index + length].copy_from_slice(value);
            }
        }
        return local_index + self.len();
    }

    #[inline]
    pub fn packed_entries_slice<
        'a,
        F,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: &StarkEvaluationVars<'a, FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> &'a [P]
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            MemorySlice::Local(index, length) => &vars.local_values[*index..*index + length],
            MemorySlice::Next(index, length) => &vars.next_values[*index..*index + length],
            _ => panic!("Cannot read from a non-local register"),
        }
    }

    #[inline]
    pub fn packed_entries<
        F,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: &StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<P>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            MemorySlice::Local(index, length) => {
                vars.local_values[*index..*index + length].to_vec()
            }
            MemorySlice::Next(index, length) => vars.next_values[*index..*index + length].to_vec(),
            MemorySlice::First(index, length) => vars.public_inputs[*index..*index + length]
                .iter()
                .map(|x| P::from(*x))
                .collect(),
            MemorySlice::Last(index, length) => vars.public_inputs[*index..*index + length]
                .iter()
                .map(|x| P::from(*x))
                .collect(),
        }
    }

    #[inline]
    pub fn evaluation_targets<
        'a,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const D: usize,
    >(
        &self,
        vars: &StarkEvaluationTargets<'a, D, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> &'a [ExtensionTarget<D>] {
        match self {
            MemorySlice::Local(index, length) => &vars.local_values[*index..*index + length],
            MemorySlice::Next(index, length) => &vars.next_values[*index..*index + length],
            MemorySlice::First(index, length) => &vars.public_inputs[*index..*index + length],
            MemorySlice::Last(index, length) => &vars.public_inputs[*index..*index + length],
        }
    }
}
