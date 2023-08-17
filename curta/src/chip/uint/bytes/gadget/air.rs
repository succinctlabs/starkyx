use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::chip::uint::bytes::lookup_table::ByteInstructionSet;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub(crate) const NUM_BYTE_GADGET_COLUMNS: usize = 103 + 51;

#[derive(Debug, Clone, Copy)]
pub struct ByteGadgetParameters<F, E, const D: usize>(PhantomData<(F, E)>);

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> const AirParameters
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
