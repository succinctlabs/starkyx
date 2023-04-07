use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

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
pub trait CircuitLayout<F: RichField + Extendable<D>, const D: usize, const N: usize>:
    Sized + Send + Sync
{
    const PUBLIC_INPUTS: usize;
    const NUM_ARITHMETIC_COLUMNS: usize;
    const ENTRY_COLUMN: usize;
    const TABLE_INDEX: usize;

    type Layouts: OpcodeLayout<F, D>;
    const OPERATIONS: [Self::Layouts; N];

}


pub const fn num_columns<
    F: RichField + Extendable<D>,
    L: CircuitLayout<F, D, N>,
    const N: usize,
    const D: usize,
>() -> usize {
    let mut num_columns = 0;
    let mut counter =0;
    while counter < N {
        num_columns += 1;
        counter += 1;
    }
    num_columns
}
