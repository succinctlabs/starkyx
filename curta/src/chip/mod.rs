use self::constraint::Constraint;
use self::instruction::Instruction;
use self::register::element::ElementRegister;
use self::table::accumulator::Accumulator;
use self::table::evaluation::Evaluation;
use self::table::lookup::Lookup;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::math::prelude::*;

pub mod air;
pub mod bool;
pub mod builder;
pub mod constraint;
pub mod ec;
pub mod field;
pub mod hash;
pub mod instruction;
pub mod register;
pub mod table;
pub mod trace;
pub mod u32;
pub mod utils;

#[const_trait]
pub trait AirParameters {
    type Field: PrimeField64;

    type CubicParams: CubicParameters<Self::Field>;

    /// The number of columns that need to be ranged-checked to range 0..num_rows
    ///
    /// If NUM_ARITHMETIC_COLUMNS > 0 is used for field operations with 2^16 bit limbs
    /// the number of rows should be 2^16.
    const NUM_ARITHMETIC_COLUMNS: usize = 0;

    /// The number of columns that are not range checked.
    const NUM_FREE_COLUMNS: usize = 0;

    const EXTENDED_COLUMNS: usize = 0;

    /// The type of instruction that the chip supports
    type Instruction: Instruction<Self::Field>;

    fn num_columns() -> usize {
        Self::NUM_ARITHMETIC_COLUMNS + Self::NUM_FREE_COLUMNS + Self::EXTENDED_COLUMNS
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
    accumulators: Vec<Accumulator<L::CubicParams>>,
    lookup_data: Vec<Lookup<L::Field, L::CubicParams, 1>>,
    evaluation_data: Vec<Evaluation<L::Field, L::CubicParams>>,
    range_table: Option<ElementRegister>,
}
