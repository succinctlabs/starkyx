//! A chip for emulating field operations
//!
//! This chip handles the range checks for the limbs, allocating table columns for input, output,
//! and witness values.
//!

use alloc::collections::BTreeMap;
use core::fmt::Debug;
use core::ops::Range;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::air::parser::AirParser;
use super::builder::InstructionId;
use super::constraint::Constraint;
use super::instruction::write::WriteInstruction;
use super::instruction::Instruction;
use super::lookup::Lookup;
use super::register::ElementRegister;
use crate::curta::stark::Stark;

/// A layout for a circuit that emulates field operations
pub trait StarkParameters<F: RichField + Extendable<D>, const D: usize>:
    Sized + Send + Sync + Clone
{
    /// The number of columns that need to be ranged-checked to range 0..num_rows
    ///
    /// If NUM_ARITHMETIC_COLUMNS > 0 is used for field operations with 2^16 bit limbs
    /// the number of rows should be 2^16.
    const NUM_ARITHMETIC_COLUMNS: usize;

    /// The number of columns that are not range checked.
    const NUM_FREE_COLUMNS: usize;

    /// Th
    type Instruction: Instruction<F, D> + Debug;
}

#[derive(Debug, Clone)]
pub struct Chip<L, F, const D: usize>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub(crate) instructions: Vec<L::Instruction>,
    pub(crate) instruction_indices: BTreeMap<InstructionId, usize>,
    pub(crate) write_instructions: Vec<WriteInstruction>,
    pub(crate) constraints: Vec<Constraint<L::Instruction, F, D>>,
    pub(crate) range_checks_idx: (usize, usize),
    pub(crate) range_data: Option<Lookup>,
    pub(crate) range_table: Option<ElementRegister>,
    pub(crate) partial_trace_index: usize,
    pub(crate) num_verifier_challenges: usize,
}

impl<L, F, const D: usize> Chip<L, F, D>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    // #[inline]
    // pub const fn table_index(&self) -> usize {
    //     self.table_index
    // }

    // #[inline]
    // pub const fn relative_table_index(&self) -> usize {
    //     self.table_index
    // }

    #[inline]
    pub const fn range_checks_idx(&self) -> (usize, usize) {
        self.range_checks_idx
    }

    #[inline]
    pub const fn num_columns_no_range_checks(&self) -> usize {
        L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS
    }

    #[inline]
    pub const fn num_range_checks(&self) -> usize {
        L::NUM_ARITHMETIC_COLUMNS
    }

    // #[inline]
    // pub const fn col_perm_index(&self, i: usize) -> usize {
    //     2 * (i - self.range_checks_idx.0) + self.table_index + 1
    // }

    // #[inline]
    // pub const fn table_perm_index(&self, i: usize) -> usize {
    //     2 * (i - self.range_checks_idx.0) + 1 + self.table_index + 1
    // }

    #[inline]
    pub const fn num_columns() -> usize {
        //1 + L::NUM_FREE_COLUMNS + 3 * L::NUM_ARITHMETIC_COLUMNS
        L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS
    }
    #[inline]
    pub const fn arithmetic_range(&self) -> Range<usize> {
        self.range_checks_idx.0..self.range_checks_idx.1
    }

    #[inline]
    pub const fn permutations_range(&self) -> Range<usize> {
        L::NUM_FREE_COLUMNS..Self::num_columns()
    }

    pub fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) {
        for consr in self.constraints.iter() {
            consr.eval(parser);
        }

        if let Some(range_data) = &self.range_data {
            range_data.eval(parser);
        }
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}

#[derive(Debug, Clone)]
pub struct ChipStark<L, F, const D: usize>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub chip: Chip<L, F, D>,
}

impl<L, F, const D: usize> ChipStark<L, F, D>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub fn new(chip: Chip<L, F, D>) -> Self {
        Self { chip }
    }
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> Stark<F, D, 2>
    for ChipStark<L, F, D>
{
    const COLUMNS: usize = Chip::<L, F, D>::num_columns();
    const PUBLIC_INPUTS: usize = 0;
    const CHALLENGES: usize = 3;

    fn round_lengths(&self) -> [usize; 2] {
        let partial_trace_index = self.chip.partial_trace_index;
        [partial_trace_index, Self::COLUMNS - partial_trace_index]
    }

    fn num_challenges(&self, round: usize) -> usize {
        match round {
            0 => 3,
            1 => 0,
            _ => unreachable!(),
        }
    }

    fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) {
        self.chip.eval(parser);
    }

    fn constraint_degree(&self) -> usize {
        self.chip.constraint_degree()
    }
}
