#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    use crate::chip::builder::shared_memory::SharedMemory;
    use crate::chip::builder::AirBuilder;
    use crate::chip::trace::writer::TraceWriter;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::uint::register::U32Register;
    use crate::chip::uint::util::u32_to_le_field_bytes;
    use crate::chip::AirParameters;
    use crate::machine::bytes::air::ByteParameters;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::plonky2::challenger::Plonky2Challenger;
    use crate::plonky2::stark::config::{
        CurtaConfig, CurtaPoseidonGoldilocksConfig, PoseidonGoldilocksStarkConfig,
    };
    use crate::plonky2::stark::Starky;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteTest;

    impl AirParameters for ByteTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 0;
    }

    #[allow(unused)]
    fn test_mult_table_byte_ops() {
        type C = CurtaPoseidonGoldilocksConfig;
        type Hasher = <C as CurtaConfig<2>>::Hasher;
        type F = GoldilocksField;
        let shared_memory = SharedMemory::new();

        let mut table_builder = AirBuilder::<ByteParameters>::init(shared_memory.clone());
        let mut builder = AirBuilder::<ByteTest>::init(shared_memory);

        let mut table = table_builder.new_byte_lookup_table();

        let mut operations = builder.byte_operations();

        let a = builder.alloc::<U32Register>();
        let b = builder.alloc::<U32Register>();
        let c = builder.bitwise_and(&a, &b, &mut operations);

        let byte_data = builder.register_byte_lookup(&mut table, operations);

        table_builder.constraint_byte_lookup_table(&table);

        let (air, trace_data) = builder.build();
        let (table_air, table_trace_data) = table_builder.build();

        let stark = Starky::new(air);
        let table_stark = Starky::new(table_air);

        let num_rows = 1 << 5;
        let config = PoseidonGoldilocksStarkConfig::standard_fast_config(num_rows);
        let table_config = PoseidonGoldilocksStarkConfig::standard_fast_config(1 << 16);

        let writer = TraceWriter::new(&trace_data, num_rows);
        let byte_writer = TraceWriter::new(&table_trace_data, 1 << 16);

        let mut challenger = Plonky2Challenger::<F, Hasher>::new();

        table.write_table_entries(&byte_writer);
        for i in 0..(1 << 16) {
            byte_writer.write_row_instructions(&table_trace_data, i);
        }

        let mut rng = rand::thread_rng();
        for i in 0..num_rows {
            let a_val = rng.gen::<u32>();
            let b_val = rng.gen::<u32>();
            writer.write(&a, &u32_to_le_field_bytes(a_val), i);
            writer.write(&b, &u32_to_le_field_bytes(b_val), i);
            writer.write_row_instructions(&trace_data, i);
        }
        let multiplicities = byte_data.get_multiplicities(&writer);
        byte_writer.write_lookup_multiplicities(table.multiplicities(), &[multiplicities]);

        todo!();
    }
}
