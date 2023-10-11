use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};

use crate::chip::uint::operations::instruction::U32Instruction;
use crate::chip::AirParameters;
use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ByteParameters;

impl AirParameters for ByteParameters {
    type Field = GoldilocksField;
    type CubicParams = GoldilocksCubicParameters;

    type Instruction = U32Instruction;

    const NUM_ARITHMETIC_COLUMNS: usize = 0;
    const NUM_FREE_COLUMNS: usize = 1;
    const EXTENDED_COLUMNS: usize = 1;
}
