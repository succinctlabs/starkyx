use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::air::{get_preprocessed_byte_trace, ByteAir, ByteParameters};
use super::stark::ByteStark;
use crate::chip::builder::AirBuilder;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::Starky;

pub(crate) const NUM_LOOKUP_ROWS: usize = 1 << 16;

pub struct BytesBuilder<L: AirParameters> {
    pub api: AirBuilder<L>,
    pub(crate) operations: ByteLookupOperations,
    pub clk: ElementRegister,
}

impl<L: AirParameters> Builder for BytesBuilder<L> {
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

    pub fn build<C: CurtaConfig<D, F = L::Field>, const D: usize>(
        self,
        num_rows: usize,
    ) -> ByteStark<L, C, D>
    where
        L::Field: RichField + Extendable<D>,
    {
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
        lookup_builder.constraint_byte_lookup_table(&lookup_table);

        let config = StarkyConfig::<C, D>::standard_fast_config(num_rows);
        let (air, trace_data) = api.build();
        let stark = Starky::new(air);

        let lookup_config = StarkyConfig::<C, D>::standard_fast_config(NUM_LOOKUP_ROWS);
        let (lookup_air, lookup_trace_data) = lookup_builder.build();
        let lookup_stark = Starky::new(ByteAir(lookup_air));

        // Get the commitment to the preprocessed byte trace.
        let lookup_writer = TraceWriter::new(&lookup_trace_data, NUM_LOOKUP_ROWS);
        // Write lookup table values
        lookup_table.write_table_entries(&lookup_writer);
        for i in 0..NUM_LOOKUP_ROWS {
            lookup_writer.write_row_instructions(&lookup_trace_data, i);
        }
        // Generate the preprocesswed trace commitment
        let lookup_preprocessed_commitment =
            get_preprocessed_byte_trace::<L::Field, L::CubicParams, C, D>(
                &lookup_writer,
                &lookup_config,
                &lookup_stark,
            );

        ByteStark {
            config,
            stark,
            air_data: trace_data,
            multiplicity_data,
            byte_trace_cap: lookup_preprocessed_commitment.merkle_tree.cap,
            lookup_config,
            lookup_stark,
            lookup_air_data: lookup_trace_data,
            lookup_table,
        }
    }
}
