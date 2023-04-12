use anyhow::{anyhow, Result};
use plonky2::field::extension::FieldExtension;
use plonky2::field::packed::PackedField;
use plonky2::iop::ext_target::ExtensionTarget;

use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Register {
    Local(usize, usize),
    Next(usize, usize),

    // Not sure if these are needed
    First(usize, usize),
    Last(usize, usize),
}

#[derive(Debug, Clone, Copy)]
pub struct U16Array {
    register: Register,
}

#[derive(Debug, Clone, Copy)]
pub enum CellType {
    U16,
    Bit,
}

pub trait DataRegister: 'static + Sized + Clone + Send + Sync {
    const CELL: Option<CellType>;

    /// Returns an element of the field
    ///
    /// Assumes register is of the correct size
    fn from_raw_register(register: Register) -> Self;

    fn into_raw_register(self) -> Register;

    fn register(&self) -> &Register;

    /// Returns an element of the field
    ///
    /// Checks that the register is of the correct size
    fn from_register(register: Register) -> Result<Self> {
        if register.len() != Self::size_of() {
            return Err(anyhow!("Invalid register length"));
        }

        Ok(Self::from_raw_register(register))
    }

    fn next(&self) -> Self {
        Self::from_raw_register(self.register().next())
    }

    fn size_of() -> usize;
}

pub struct WitnessData {
    size: usize,
    cell_type: Option<CellType>,
}

impl WitnessData {
    pub fn u16(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: Some(CellType::U16),
        }
    }

    pub fn bitarray(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: Some(CellType::Bit),
        }
    }

    pub fn untyped(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: None,
        }
    }

    pub fn destruct(self) -> (usize, Option<CellType>) {
        (self.size, self.cell_type)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BitArray<const N: usize> {
    register: Register,
}

#[derive(Debug, Clone, Copy)]
pub struct BitRegister {
    register: Register,
}

impl Register {
    #[inline]
    pub fn is_next(&self) -> bool {
        match self {
            Register::Next(_, _) => true,
            _ => false,
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Register::Local(index, length) => Register::Next(*index, *length),
            _ => panic!("Invalid register type for the next register"),
        }
    }

    #[inline]
    pub const fn get_range(&self) -> (usize, usize) {
        match self {
            Register::Local(index, length) => (*index, *index + length),
            Register::Next(index, length) => (*index, *index + length),
            Register::First(index, length) => (*index, *index + length),
            Register::Last(index, length) => (*index, *index + length),
        }
    }

    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            Register::Local(index, _) => *index,
            Register::Next(index, _) => *index,
            Register::First(index, _) => *index,
            Register::Last(index, _) => *index,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        match self {
            Register::Local(_, length) => *length,
            Register::Next(_, length) => *length,
            Register::First(_, length) => *length,
            Register::Last(_, length) => *length,
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
                value.copy_from_slice(&trace_rows[row_index][*index..*index + length]);
            }
            Register::Next(index, length) => {
                value.copy_from_slice(&trace_rows[row_index + 1][*index..*index + length]);
            }
            _ => panic!("Cannot read from a non-local register"),
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
            Register::First(index, length) => {
                trace_rows[0][*index..*index + length].copy_from_slice(value);
            }
            Register::Last(index, length) => {
                trace_rows[trace_rows.len() - 1][*index..*index + length].copy_from_slice(value);
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
            Register::Local(index, length) => vars.local_values[*index..*index + length].to_vec(),
            Register::Next(index, length) => vars.next_values[*index..*index + length].to_vec(),
            Register::First(index, length) => vars.public_inputs[*index..*index + length]
                .iter()
                .map(|x| P::from(*x))
                .collect(),
            Register::Last(index, length) => vars.public_inputs[*index..*index + length]
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
            Register::Local(index, length) => &vars.local_values[*index..*index + length],
            Register::Next(index, length) => &vars.next_values[*index..*index + length],
            Register::First(index, length) => &vars.public_inputs[*index..*index + length],
            Register::Last(index, length) => &vars.public_inputs[*index..*index + length],
        }
    }
}

impl DataRegister for BitRegister {
    const CELL: Option<CellType> = Some(CellType::Bit);

    fn from_raw_register(register: Register) -> Self {
        Self { register }
    }

    fn register(&self) -> &Register {
        &self.register
    }

    fn size_of() -> usize {
        1
    }

    fn into_raw_register(self) -> Register {
        self.register
    }
}

impl<const N: usize> DataRegister for BitArray<N> {
    const CELL: Option<CellType> = Some(CellType::Bit);

    fn from_raw_register(register: Register) -> Self {
        Self { register }
    }

    fn register(&self) -> &Register {
        &self.register
    }

    fn size_of() -> usize {
        N
    }

    fn into_raw_register(self) -> Register {
        self.register
    }
}

impl DataRegister for U16Array {
    const CELL: Option<CellType> = Some(CellType::U16);

    fn from_raw_register(register: Register) -> Self {
        Self { register }
    }

    fn register(&self) -> &Register {
        &self.register
    }

    fn size_of() -> usize {
        // TODO: FIX
        32
    }

    fn into_raw_register(self) -> Register {
        self.register
    }
}
