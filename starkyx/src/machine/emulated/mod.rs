use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::chip::instruction::empty::EmptyInstruction;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub mod builder;
pub mod proof;
pub mod stark;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RangeParameters<F, E>(pub PhantomData<(F, E)>);

impl<F: PrimeField64, E: CubicParameters<F>> AirParameters for RangeParameters<F, E> {
    type Field = F;
    type CubicParams = E;

    type Instruction = EmptyInstruction<F>;

    const NUM_ARITHMETIC_COLUMNS: usize = 0;
    const NUM_FREE_COLUMNS: usize = 2;
    const EXTENDED_COLUMNS: usize = 6;
}
