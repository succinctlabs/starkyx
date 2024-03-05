use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use self::constraint::Constraint;
use self::instruction::Instruction;
use crate::math::prelude::*;
use crate::plonky2::stark::Starky;

pub mod air;
pub mod arithmetic;
pub mod bool;
pub mod builder;
pub mod constraint;
pub mod ec;
pub mod field;
pub mod instruction;
pub mod memory;
pub mod register;
pub mod table;
pub mod trace;
pub mod uint;
pub mod utils;

use core::fmt::Debug;
pub trait AirParameters:
    'static + Clone + Send + Sync + Sized + Debug + Serialize + DeserializeOwned
{
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

    /// a unique identifier for the air parameters.
    ///
    /// by default, this method uses the type name of the air parameters. In case the Rust
    /// 'TypeId' is not functioning properly, this method should be overridden.
    fn id() -> String {
        format!("{:?}", std::any::TypeId::of::<Self>()).to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Chip<L: AirParameters> {
    constraints: Vec<Constraint<L>>,
    global_constraints: Vec<Constraint<L>>,
    pub execution_trace_length: usize,
    pub num_challenges: usize,
    pub num_public_values: usize,
    pub num_global_values: usize,
}

impl<L: AirParameters> Starky<Chip<L>> {
    pub fn from_chip(chip: Chip<L>) -> Self {
        Self::new(chip)
    }
}
