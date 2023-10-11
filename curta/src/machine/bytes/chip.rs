use super::air::ByteParameters;
use crate::chip::builder::shared_memory::SharedMemory;
use crate::chip::builder::AirBuilder;
use crate::chip::trace::data::AirTraceData;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::lookup_table::multiplicity_data::ByteMultiplicityData;
use crate::chip::Chip;

pub struct ByteBuilder {
    builder: AirBuilder<ByteParameters>,
    byte_operations: ByteLookupOperations,
}

// impl ByteChip {
//     pub fn init(shared_memory: SharedMemory) -> Self {

//     }
// }

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use serde::{Deserialize, Serialize};

    use crate::chip::builder::shared_memory::SharedMemory;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::AirParameters;
    use crate::machine::bytes::air::ByteParameters;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

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

    #[test]
    fn test_mult_table_byte_ops() {
        let shared_memory = SharedMemory::new();

        let mut table_builder = AirBuilder::<ByteParameters>::init(shared_memory.clone());
        let mut builder = AirBuilder::<ByteTest>::init(shared_memory);

        let table_operations = table_builder.byte_operations();
    }
}
