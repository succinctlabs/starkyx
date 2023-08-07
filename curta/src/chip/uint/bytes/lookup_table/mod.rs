use std::sync::mpsc;

use self::builder_operations::ByteLookupOperations;
use self::table::ByteLookupTable;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::NUM_CHALLENGES;
use crate::chip::AirParameters;

pub mod builder_operations;
pub mod multiplicity_data;
pub mod table;

impl<L: AirParameters> AirBuilder<L> {
    pub fn byte_operations(&mut self) -> (ByteLookupOperations, ByteLookupTable<L::Field>) {
        let (tx, rx) = mpsc::channel::<ByteOperation<u8>>();

        let row_acc_challenges = self.alloc_challenge_array::<CubicRegister>(NUM_CHALLENGES);

        let lookup_table = self.new_byte_lookup_table(row_acc_challenges, rx);
        let operations = ByteLookupOperations::new(tx, row_acc_challenges);

        (operations, lookup_table)
    }

    pub fn register_byte_lookup(
        &mut self,
        operation_values: ByteLookupOperations,
        table: &ByteLookupTable<L::Field>,
    ) {
        let multiplicities = table.multiplicity_data.multiplicities().clone();
        let lookup_challenge = self.alloc_challenge::<CubicRegister>();

        let lookup_table = self.lookup_table_with_multiplicities(
            &lookup_challenge,
            &table.digests,
            &multiplicities,
        );
        let lookup_values = self.lookup_values(&lookup_challenge, &operation_values.values);

        self.cubic_lookup_from_table_and_values(lookup_table, lookup_values);
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::register::Register;
    use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
    use crate::chip::uint::bytes::register::ByteRegister;
    use crate::chip::AirParameters;
    use crate::plonky2::field::Field;

    #[derive(Debug, Clone)]
    struct ByteOpTest;

    impl const AirParameters for ByteOpTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ByteOperationInstruction;

        const NUM_FREE_COLUMNS: usize = 429;
        const EXTENDED_COLUMNS: usize = 582;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_bit_op_lookup() {
        type F = GoldilocksField;
        type L = ByteOpTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let num_ops = 120;

        let mut builder = AirBuilder::<L>::new();

        let (mut operations, mut table) = builder.byte_operations();

        let mut a_vec = Vec::new();
        let mut b_vec = Vec::new();

        for _ in 0..num_ops {
            let a = builder.alloc::<ByteRegister>();
            let b = builder.alloc::<ByteRegister>();
            let result = builder.alloc::<ByteRegister>();
            let op = ByteOperation::And(a, b, result);
            builder.set_byte_operation(&op, &mut operations);
            a_vec.push(a);
            b_vec.push(b);
        }

        builder.register_byte_lookup(operations, &table);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        let mut rng = thread_rng();
        for i in 0..L::num_rows() {
            let a_v = rng.gen::<u8>();
            let b_v = rng.gen::<u8>();

            for k in 0..num_ops {
                writer.write(&a_vec[k], &F::from_canonical_u8(a_v), i);
                writer.write(&b_vec[k], &F::from_canonical_u8(b_v), i);
            }

            writer.write_row_instructions(&air, i);
        }

        table.write(&writer, L::num_rows() * 2);

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
