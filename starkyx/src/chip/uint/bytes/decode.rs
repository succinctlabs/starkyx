use serde::{Deserialize, Serialize};

use super::register::ByteRegister;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::ConstraintInstruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::Register;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ByteDecodeInstruction {
    byte: ByteRegister,
    bits: ArrayRegister<BitRegister>,
}

impl ByteDecodeInstruction {
    pub fn new(byte: ByteRegister, bits: ArrayRegister<BitRegister>) -> Self {
        Self { byte, bits }
    }
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn decode_byte(&mut self, byte: &ByteRegister, bits: &ArrayRegister<BitRegister>)
    where
        L::Instruction: From<ByteDecodeInstruction>,
    {
        let instruction = ByteDecodeInstruction::new(*byte, *bits);
        self.register_instruction(instruction);
    }
}

impl<AP: AirParser> AirConstraint<AP> for ByteDecodeInstruction {
    fn eval(&self, parser: &mut AP) {
        let byte = self.byte.eval(parser);
        let bits = self.bits.eval_array::<_, 8>(parser);

        let mut acc = parser.zero();
        for (i, bit) in bits.into_iter().enumerate() {
            let two_i = parser.constant(AP::Field::from_canonical_u32(1 << i as u32));
            let two_i_bit = parser.mul(two_i, bit);
            acc = parser.add(acc, two_i_bit);
        }
        parser.assert_eq(byte, acc);
    }
}

impl ConstraintInstruction for ByteDecodeInstruction {}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct DecodeTest;

    impl AirParameters for DecodeTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ByteDecodeInstruction;

        const NUM_FREE_COLUMNS: usize = 16;
    }

    #[test]
    fn test_byte_decode() {
        type F = GoldilocksField;
        type L = DecodeTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let byte = builder.alloc::<ByteRegister>();
        let bits = builder.alloc_array::<BitRegister>(8);

        builder.decode_byte(&byte, &bits);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 9;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let writer = generator.new_writer();
        let mut rng = thread_rng();
        for i in 0..num_rows {
            let byte_val: u8 = rng.gen();
            writer.write(&byte, &F::from_canonical_u8(byte_val), i);
            for (j, bit) in bits.into_iter().enumerate() {
                let bit_val = (byte_val >> j) & 1;
                writer.write(&bit, &F::from_canonical_u8(bit_val), i);
            }
        }
        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
