use super::stark::EmulatedStark;
use super::RangeParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::RegisterSerializable;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::Starky;

pub(crate) const NUM_LOOKUP_ROWS: usize = 1 << 16;

pub struct EmulatedBuilder<L: AirParameters> {
    pub api: AirBuilder<L>,
    pub clk: ElementRegister,
}

impl<L: AirParameters> Builder for EmulatedBuilder<L> {
    type Field = L::Field;
    type CubicParams = L::CubicParams;
    type Parameters = L;
    type Instruction = L::Instruction;

    fn api(&mut self) -> &mut AirBuilder<Self::Parameters> {
        &mut self.api
    }

    fn clk(&mut self) -> ElementRegister {
        self.clk
    }
}

impl<L: AirParameters> EmulatedBuilder<L> {
    pub fn new() -> Self {
        let mut api = AirBuilder::<L>::new();
        let clk = api.clock();
        api.internal_range_check = false;
        api.init_local_memory();
        EmulatedBuilder { api, clk }
    }

    pub fn build<C: CurtaConfig<D, F = L::Field>, const D: usize>(
        self,
        num_rows: usize,
    ) -> EmulatedStark<L, C, D> {
        let EmulatedBuilder { mut api, .. } = self;
        let shared_memory = api.shared_memory.clone();
        let mut lookup_builder =
            AirBuilder::<RangeParameters<L::Field, L::CubicParams>>::init(shared_memory);

        // Lookup table entry.
        let lookup_table = lookup_builder.clock();
        // Allocate multiplicities.
        let multiplicity = lookup_builder.alloc_array::<ElementRegister>(1);

        let values = ArrayRegister::<ElementRegister>::from_register_unsafe(MemorySlice::Local(
            0,
            L::NUM_ARITHMETIC_COLUMNS,
        ))
        .into_iter()
        .chain(api.global_arithmetic.iter().copied())
        .collect::<Vec<_>>();

        let mut table_data = lookup_builder.new_lookup(&[lookup_table], &multiplicity);
        let lookup_values = table_data.register_lookup_values(&mut api, &values);
        lookup_builder.constrain_element_lookup_table(table_data);

        let config = StarkyConfig::<C, D>::standard_fast_config(num_rows);
        let (air, trace_data) = api.build();
        let stark = Starky::new(air);

        let lookup_config = StarkyConfig::<C, D>::standard_fast_config(NUM_LOOKUP_ROWS);
        let (lookup_air, lookup_trace_data) = lookup_builder.build();
        let lookup_stark = Starky::new(lookup_air);

        EmulatedStark {
            config,
            stark,
            air_data: trace_data,
            lookup_config,
            lookup_stark,
            lookup_air_data: lookup_trace_data,
            lookup_values,
            lookup_table,
            multiplicity,
        }
    }
}
