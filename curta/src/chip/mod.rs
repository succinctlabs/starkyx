use self::instruction::Instruction;
use crate::air::parser::{AirParser, MulParser};
use crate::air::AirConstraint;
use crate::math::prelude::*;

pub mod air;
pub mod builder;
pub mod constraint;
pub mod ec;
pub mod field;
pub mod instruction;
pub mod register;

pub trait AirParameters {
    type Field: Field;

    /// The number of columns that need to be ranged-checked to range 0..num_rows
    ///
    /// If NUM_ARITHMETIC_COLUMNS > 0 is used for field operations with 2^16 bit limbs
    /// the number of rows should be 2^16.
    const NUM_ARITHMETIC_COLUMNS: usize;

    /// The number of columns that are not range checked.
    const NUM_FREE_COLUMNS: usize;

    /// The type of instruction that the chip supports
    type Instruction: Instruction<Self::Field>;
}

pub trait Parser<L: AirParameters>: AirParser
where
    L::Instruction: AirConstraint<Self> + for<'a> AirConstraint<MulParser<'a, Self>>,
{
}

pub struct Chip<L: AirParameters> {
    constraints: Vec<L::Instruction>,
}
