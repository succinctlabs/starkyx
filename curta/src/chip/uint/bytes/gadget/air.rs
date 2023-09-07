use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use serde::{Deserialize, Serialize};

use crate::chip::uint::bytes::lookup_table::ByteInstructionSet;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ByteGadgetParameters<F, E, const D: usize>(PhantomData<(F, E)>);

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> AirParameters
    for ByteGadgetParameters<F, E, D>
{
    type Field = F;
    type CubicParams = E;

    const NUM_FREE_COLUMNS: usize = 103;
    const EXTENDED_COLUMNS: usize = 51;

    type Instruction = ByteInstructionSet;

    fn num_rows_bits() -> usize {
        16
    }
}
