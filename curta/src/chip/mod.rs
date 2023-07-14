use self::constraint::Constraint;
use self::instruction::Instruction;
use self::register::element::ElementRegister;
use self::table::lookup::Lookup;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::math::prelude::*;

pub mod air;
pub mod bool;
pub mod builder;
pub mod constraint;
pub mod ec;
pub mod field;
pub mod instruction;
pub mod register;
pub mod table;
pub mod trace;
pub mod utils;

#[const_trait]
pub trait AirParameters {
    type Field: PrimeField64;

    type CubicParams: CubicParameters<Self::Field>;

    /// The number of columns that need to be ranged-checked to range 0..num_rows
    ///
    /// If NUM_ARITHMETIC_COLUMNS > 0 is used for field operations with 2^16 bit limbs
    /// the number of rows should be 2^16.
    const NUM_ARITHMETIC_COLUMNS: usize;

    /// The number of columns that are not range checked.
    const NUM_FREE_COLUMNS: usize;

    /// The type of instruction that the chip supports
    type Instruction: Instruction<Self::Field>;

    fn num_columns() -> usize {
        Self::NUM_ARITHMETIC_COLUMNS + Self::NUM_FREE_COLUMNS
    }

    fn num_rows_bits() -> usize;

    fn num_rows() -> usize {
        1 << Self::num_rows_bits()
    }
}

#[derive(Debug, Clone)]
pub struct Chip<L: AirParameters> {
    constraints: Vec<Constraint<L>>,
    execution_trace_length: usize,
    num_challenges: usize,
    lookup_data: Vec<Lookup<L::Field, L::CubicParams, 1>>,
    range_table: Option<ElementRegister>,
}
