use super::air::ByteParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::trace::data::AirTraceData;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::{AirParameters, Chip};
use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::Starky;

pub struct ByteStark<L: AirParameters> {
    stark: Starky<Chip<L>>,
    air_data: AirTraceData<L>,
    lookup_stark: Starky<Chip<ByteParameters<L::Field, L::CubicParams>>>,
    lookup_trace_data: AirTraceData<ByteParameters<L::Field, L::CubicParams>>,
}
