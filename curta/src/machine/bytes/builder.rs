use super::air::ByteParameters;
use super::stark::ByteStark;
use crate::chip::builder::AirBuilder;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::ByteArrayRegister;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::Starky;

pub(crate) const NUM_LOOKUP_ROWS: usize = 1 << 16;

pub struct BytesBuilder<L: AirParameters> {
    pub api: AirBuilder<L>,
    operations: ByteLookupOperations,
    pub clk: ElementRegister,
}

impl<L: AirParameters> Builder for BytesBuilder<L> {
    type Field = L::Field;
    type Parameters = L;

    fn api(&mut self) -> &mut AirBuilder<Self::Parameters> {
        &mut self.api
    }
}

impl<L: AirParameters> BytesBuilder<L>
where
    L::Instruction: UintInstructions,
{
    pub fn new() -> Self {
        let mut api = AirBuilder::<L>::new();
        let clk = api.clock();
        api.init_local_memory();
        BytesBuilder {
            api,
            operations: ByteLookupOperations::new(),
            clk,
        }
    }

    pub fn bitwise_and<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        b: &ByteArrayRegister<N>,
    ) -> ByteArrayRegister<N> {
        self.api.bitwise_and(a, b, &mut self.operations)
    }

    pub fn build<C: CurtaConfig<D, F = L::Field>, const D: usize>(
        self,
        num_rows: usize,
    ) -> ByteStark<L, C, D> {
        let BytesBuilder {
            mut api,
            operations,
            ..
        } = self;
        let shared_memory = api.shared_memory.clone();
        let mut lookup_builder =
            AirBuilder::<ByteParameters<L::Field, L::CubicParams>>::init(shared_memory);

        let mut lookup_table = lookup_builder.new_byte_lookup_table();
        let multiplicity_data = api.register_byte_lookup(&mut lookup_table, operations);

        let config = StarkyConfig::<C, D>::standard_fast_config(num_rows);
        let (air, trace_data) = api.build();
        let stark = Starky::new(air);

        let lookup_config = StarkyConfig::<C, D>::standard_fast_config(NUM_LOOKUP_ROWS);
        let (lookup_air, lookup_trace_data) = lookup_builder.build();
        let lookup_stark = Starky::new(lookup_air);

        ByteStark {
            config,
            stark,
            air_data: trace_data,
            multiplicity_data,
            lookup_config,
            lookup_stark,
            lookup_air_data: lookup_trace_data,
            lookup_table,
        }
    }
}
