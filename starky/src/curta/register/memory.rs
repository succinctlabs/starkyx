use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;

use crate::curta::constraint::arithmetic::{ArithmeticExpression, ArithmeticExpressionSlice};
use crate::curta::new_stark::vars as new_vars;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

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
        return local_index + self.len();
    }

    #[inline]
    pub fn packed_generic_vars<
        'a,
        F,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<'a, FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> &'a [P]
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            MemorySlice::Local(index, length) => &vars.local_values[*index..*index + length],
            MemorySlice::Next(index, length) => &vars.next_values[*index..*index + length],
            _ => unimplemented!("Cannot get generic vars for public inputs or challenges"),
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
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
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
            MemorySlice::Public(index, length) => {
                vars.public_inputs.map(|x| P::from(x))[*index..*index + length].into()
            }
            MemorySlice::Challenge(_, _) => unimplemented!("Cannot get entries for challenges"),
        }
    }

    #[inline]
    pub fn packed_generic_vars_new<
        'a,
        F,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        vars: new_vars::StarkEvaluationVars<
            'a,
            FE,
            P,
            { COLUMNS },
            { PUBLIC_INPUTS },
            { CHALLENGES },
        >,
    ) -> &'a [P]
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            MemorySlice::Local(index, length) => &vars.local_values[*index..*index + length],
            MemorySlice::Next(index, length) => &vars.next_values[*index..*index + length],
            _ => unimplemented!("Cannot get generic vars for public inputs or challenges"),
        }
    }

    #[inline]
    pub fn packed_entries_new<
        F,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        vars: new_vars::StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
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
            MemorySlice::Public(index, length) => {
                vars.public_inputs.map(|x| P::from(x))[*index..*index + length].into()
            }
            MemorySlice::Challenge(index, length) => {
                vars.challenges.map(|x| P::from(x))[*index..*index + length].into()
            }
        }
    }

    #[inline]
    pub fn ext_circuit_vars<
        'a,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const D: usize,
    >(
        &self,
        vars: StarkEvaluationTargets<'a, D, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> &'a [ExtensionTarget<D>] {
        match self {
            MemorySlice::Local(index, length) => &vars.local_values[*index..*index + length],
            MemorySlice::Next(index, length) => &vars.next_values[*index..*index + length],
            MemorySlice::Public(index, length) => &vars.public_inputs[*index..*index + length],
            MemorySlice::Challenge(_, _) => unimplemented!("Cannot get entries for challenges"),
        }
    }

    #[inline]
    pub fn ext_circuit_vars_new<
        'a,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
        const D: usize,
    >(
        &self,
        vars: new_vars::StarkEvaluationTargets<
            'a,
            D,
            { COLUMNS },
            { PUBLIC_INPUTS },
            { CHALLENGES },
        >,
    ) -> &'a [ExtensionTarget<D>] {
        match self {
            MemorySlice::Local(index, length) => &vars.local_values[*index..*index + length],
            MemorySlice::Next(index, length) => &vars.next_values[*index..*index + length],
            MemorySlice::Public(index, length) => &vars.public_inputs[*index..*index + length],
            MemorySlice::Challenge(index, length) => &vars.challenges[*index..*index + length],
        }
    }

    pub fn expr<F: RichField + Extendable<D>, const D: usize>(&self) -> ArithmeticExpression<F, D> {
        ArithmeticExpression {
            expression: ArithmeticExpressionSlice::from_raw_register(*self),
            size: self.len(),
        }
    }
}
