use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::instruction::Instruction;
use super::layout::OpcodeLayout;

/// A layout for a circuit that emulates field operations
pub trait EmulatedCircuitLayout<F: RichField + Extendable<D>, const D: usize, const N: usize>:
    Sized + Send + Sync
{
    const PUBLIC_INPUTS: usize;
    const NUM_ARITHMETIC_COLUMNS: usize;
    const ENTRY_COLUMN: usize;
    const TABLE_INDEX: usize;

    type Layouts: OpcodeLayout<F, D>;
    const OPERATIONS: [Self::Layouts; N];

    /// Check that the operations allocations are consistent with total number of columns
    fn is_consistent(&self) -> bool {
        assert_eq!(
            Self::TABLE_INDEX,
            Self::ENTRY_COLUMN + Self::NUM_ARITHMETIC_COLUMNS
        );
        true
    }
}

/// A layout for a circuit that emulates field operations
pub trait StarkParameters<F: RichField + Extendable<D>, const D: usize>:
    Sized + Send + Sync
{
    const PUBLIC_INPUTS: usize;
    const NUM_ARITHMETIC_COLUMNS: usize;
    const NUM_STARK_COLUMNS: usize;

    type Instruction: Instruction<F, D>;
}

/// A layout for a circuit that emulates field operations
pub trait ChipParameters<F: RichField + Extendable<D>, const D: usize>:
    Sized + Send + Sync
{
    const NUM_ARITHMETIC_COLUMNS: usize;
    const NUM_FREE_COLUMNS: usize;

    type Input: Clone + Send + Sync;
    type Output: Clone + Send + Sync;

    type Instruction: Instruction<F, D>;
}

impl<F: RichField + Extendable<D>, const D: usize, T: StarkParameters<F, D>> ChipParameters<F, D>
    for T
{
    const NUM_ARITHMETIC_COLUMNS: usize = Self::NUM_ARITHMETIC_COLUMNS;
    const NUM_FREE_COLUMNS: usize = Self::NUM_STARK_COLUMNS;

    type Input = ();
    type Output = ();

    type Instruction = <Self as StarkParameters<F, D>>::Instruction;
}
