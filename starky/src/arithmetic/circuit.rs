use plonky2::field::extension::FieldExtension;
use plonky2::field::packed::PackedField;
use plonky2::iop::ext_target::ExtensionTarget;

use anyhow::{Result, anyhow};

use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub enum Register {
    Local(usize, usize),
    Next(usize, usize),
}



#[derive(Debug, Clone, Copy)]
pub enum CellType {
    U16,
    Bit
}

pub trait DataRegister : 'static + Sized + Send + Sync {
    const CELL : CellType;

    /// Returns an element of the field
    /// 
    /// Assumes register is of the correct size
    fn from_raw_register(register : Register) -> Self;

    /// Returns an element of the field
    /// 
    /// Checks that the register is of the correct size
    fn from_register(register : Register) -> Result<Self> {
        if register.len() != Self::size_of() {
            return Err(anyhow!("Invalid register length"));
        }

        Ok(Self::from_raw_register(register))
    }

    fn size_of() -> usize;
}

pub trait WitnessRegister {}


impl Register {
    #[inline]
    pub const fn get_range(&self) -> (usize, usize) {
        match self {
            Register::Local(index, length) => (*index, *index + length),
            Register::Next(index, length) => (*index, *index + length),
        }
    }

    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            Register::Local(index, _) => *index,
            Register::Next(index, _) => *index,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        match self {
            Register::Local(_, length) => *length,
            Register::Next(_, length) => *length,
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn read<T: Copy>(&self, trace_rows: &mut [Vec<T>], value: &mut [T], row_index: usize) {
        match self {
            Register::Local(index, length) => {
                value.copy_from_slice(&mut trace_rows[row_index][*index..*index + length]);
            }
            Register::Next(index, length) => {
                value.copy_from_slice(&mut trace_rows[row_index + 1][*index..*index + length]);
            }
        }
    }

    #[inline]
    pub fn assign<T: Copy>(&self, trace_rows: &mut [Vec<T>], value: &mut [T], row_index: usize) {
        match self {
            Register::Local(index, length) => {
                trace_rows[row_index][*index..*index + length].copy_from_slice(value);
            }
            Register::Next(index, length) => {
                trace_rows[row_index + 1][*index..*index + length].copy_from_slice(value);
            }
        }
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
            Register::Local(index, length) => &vars.local_values[*index..*index + length],
            Register::Next(index, length) => &vars.next_values[*index..*index + length],
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
            Register::Local(index, length) => &vars.local_values[*index..*index + length],
            Register::Next(index, length) => &vars.next_values[*index..*index + length],
        }
    }
}


pub struct EmulatedASM {
    registers: Vec<Register>,
    trace_rows: Vec<Vec<u64>>,
    current_row: usize,
}

impl EmulatedASM {

}