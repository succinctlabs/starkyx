use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::air::ByteParameters;
use super::proof::ByteStarkProof;
use crate::chip::trace::data::AirTraceData;
use crate::chip::uint::bytes::lookup_table::multiplicity_data::ByteMultiplicityData;
use crate::chip::uint::bytes::lookup_table::table::ByteLogLookupTable;
use crate::chip::{AirParameters, Chip};
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::Starky;
use crate::plonky2::Plonky2Air;
use crate::trace::AirTrace;

pub struct ByteStark<L: AirParameters, C, const D: usize> {
    pub(crate) config: StarkyConfig<C, D>,
    pub(crate) stark: Starky<Chip<L>>,
    pub(crate) air_data: AirTraceData<L>,
    pub(crate) multiplicity_data: ByteMultiplicityData,
    pub(crate) lookup_config: StarkyConfig<C, D>,
    pub(crate) lookup_stark: Starky<Chip<ByteParameters<L::Field, L::CubicParams>>>,
    pub(crate) lookup_air_data: AirTraceData<ByteParameters<L::Field, L::CubicParams>>,
    pub(crate) lookup_table: ByteLogLookupTable<L::Field, L::CubicParams>,
}

pub struct ByteTrace<F> {
    execusion_trace: AirTrace<F>,
    extended_trace: AirTrace<F>,
    lookup_trace: AirTrace<F>,
    lookup_extended_trace: AirTrace<F>,
}

impl<L: AirParameters, C: CurtaConfig<D, F = L::Field>, const D: usize> ByteStark<L, C, D>
where
    L::Field: RichField + Extendable<D>,
    Chip<L>: Plonky2Air<L::Field, D>,
{
    fn generate_trace(&self, execusion_trace: AirTrace<L::Field>) -> ByteTrace<L::Field> {
        todo!()
    }

    pub fn prove(&self, execusion_trace: AirTrace<L::Field>) -> ByteStarkProof<L::Field, C, D> {
        let trace = self.generate_trace(execusion_trace);
        todo!()
    }
}
