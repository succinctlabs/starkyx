use super::air::ByteParameters;
use super::stark::ByteStark;
use crate::chip::builder::AirBuilder;
use crate::chip::trace::data::AirTraceData;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::{AirParameters, Chip};
use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::Starky;

pub struct BytesBuilder<L: AirParameters> {
    pub api: AirBuilder<L>,
    pub(crate) operations: ByteLookupOperations,
}

impl<L: AirParameters> BytesBuilder<L>
where
    L: UintInstructions,
{
    pub fn new() -> Self {
        let api = AirBuilder::<L>::new();
        BytesBuilder {
            api,
            operations: ByteLookupOperations::new(),
        }
    }

    pub fn build<C: CurtaConfig<D, F = L::Field>, const D: usize>(self) -> ByteStark<L> {
        unimplemented!()
    }
}
