use std::sync::mpsc;

use self::builder_operations::ByteLookupOperations;
use self::table::ByteLookupTable;
use super::bit_operations::and::And;
use super::bit_operations::not::Not;
use super::bit_operations::xor::Xor;
use super::decode::ByteDecodeInstruction;
use super::operations::instruction::ByteOperationInstruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::bool::SelectInstruction;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::NUM_CHALLENGES;
use crate::chip::AirParameters;

pub mod builder_operations;
pub mod multiplicity_data;
pub mod table;

use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum ByteInstructionSet {
    Op(ByteOperationInstruction),
    BitAnd(And<8>),
    BitXor(Xor<8>),
    BitNot(Not<8>),
    BitSelect(SelectInstruction<BitRegister>),
    Decode(ByteDecodeInstruction),
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn byte_operations(&mut self) -> (ByteLookupOperations, ByteLookupTable<L::Field>)
    where
        L::Instruction: From<ByteInstructionSet> + From<SelectInstruction<BitRegister>> + From<ByteDecodeInstruction>,
    {
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

impl<AP: AirParser> AirConstraint<AP> for ByteInstructionSet {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Op(op) => op.eval(parser),
            Self::BitAnd(op) => op.eval(parser),
            Self::BitXor(op) => op.eval(parser),
            Self::BitNot(op) => op.eval(parser),
            Self::BitSelect(op) => op.eval(parser),
            Self::Decode(instruction) => instruction.eval(parser),
        }
    }
}

impl<F: PrimeField64> Instruction<F> for ByteInstructionSet {
    fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            Self::Op(op) => Instruction::<F>::inputs(op),
            Self::BitAnd(op) => Instruction::<F>::inputs(op),
            Self::BitXor(op) => Instruction::<F>::inputs(op),
            Self::BitNot(op) => Instruction::<F>::inputs(op),
            Self::BitSelect(op) => Instruction::<F>::inputs(op),
            Self::Decode(instruction) => Instruction::<F>::inputs(instruction),
        }
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            Self::Op(op) => Instruction::<F>::trace_layout(op),
            Self::BitAnd(op) => Instruction::<F>::trace_layout(op),
            Self::BitXor(op) => Instruction::<F>::trace_layout(op),
            Self::BitNot(op) => Instruction::<F>::trace_layout(op),
            Self::BitSelect(op) => Instruction::<F>::trace_layout(op),
            Self::Decode(instruction) => Instruction::<F>::trace_layout(instruction),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Op(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitAnd(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitXor(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitNot(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitSelect(op) => Instruction::<F>::write(op, writer, row_index),
            Self::Decode(instruction) => Instruction::<F>::write(instruction, writer, row_index),
        }
    }
}

impl From<ByteOperationInstruction> for ByteInstructionSet {
    fn from(op: ByteOperationInstruction) -> Self {
        Self::Op(op)
    }
}

impl From<And<8>> for ByteInstructionSet {
    fn from(op: And<8>) -> Self {
        Self::BitAnd(op)
    }
}

impl From<Xor<8>> for ByteInstructionSet {
    fn from(op: Xor<8>) -> Self {
        Self::BitXor(op)
    }
}

impl From<Not<8>> for ByteInstructionSet {
    fn from(op: Not<8>) -> Self {
        Self::BitNot(op)
    }
}

impl From<SelectInstruction<BitRegister>> for ByteInstructionSet {
    fn from(op: SelectInstruction<BitRegister>) -> Self {
        Self::BitSelect(op)
    }
}

impl From<ByteDecodeInstruction> for ByteInstructionSet {
    fn from(instruction: ByteDecodeInstruction) -> Self {
        Self::Decode(instruction)
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::bytes::register::ByteRegister;
    use crate::chip::AirParameters;
    use crate::plonky2::field::Field;

    #[derive(Debug, Clone)]
    struct ByteOpTest<const N: usize>;

    impl<const N: usize> const AirParameters for ByteOpTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ByteInstructionSet;

        const NUM_FREE_COLUMNS: usize = 3 * N + 130;
        const EXTENDED_COLUMNS: usize = 2 * N + (3 * N) / 2 + 62;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_bit_op_lookup() {
        type F = GoldilocksField;
        const NUM_OPS: usize = 4;
        type L = ByteOpTest<NUM_OPS>;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let (mut operations, mut table) = builder.byte_operations();

        let mut a_vec = Vec::new();
        let mut b_vec = Vec::new();

        for _ in 0..NUM_OPS {
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

        table.write_table_entries(&writer);

        let mut rng = thread_rng();
        for i in 0..L::num_rows() {
            let a_v = rng.gen::<u8>();
            let b_v = rng.gen::<u8>();

            for k in 0..NUM_OPS {
                writer.write(&a_vec[k], &F::from_canonical_u8(a_v), i);
                writer.write(&b_vec[k], &F::from_canonical_u8(b_v), i);
            }

            writer.write_row_instructions(&air, i);
        }

        table.write_multiplicities(&writer, L::num_rows() * 2);

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
