use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use super::EllipticCurve;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::AirParameters;
use crate::math::field::{Field, PrimeField64};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ECScalarRegister<E> {
    pub limbs: ArrayRegister<ElementRegister>,
    _marker: PhantomData<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimbBitInstruction {
    bit: BitRegister,
    bit_accumulator: ElementRegister,
    limb: ElementRegister,
    end_bit: BitRegister,
    start_bit: BitRegister,
}

impl<E: EllipticCurve> ECScalarRegister<E> {
    pub const fn new(limbs: ArrayRegister<ElementRegister>) -> Self {
        Self {
            limbs,
            _marker: PhantomData,
        }
    }
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn bit_decomposition(
        &mut self,
        limb: ElementRegister,
        start_bit: BitRegister,
        end_bit: BitRegister,
    ) -> BitRegister
    where
        L::Instruction: From<LimbBitInstruction>,
    {
        let bit_accumulator = self.alloc();
        let bit = self.alloc();

        let instruction = LimbBitInstruction {
            bit,
            bit_accumulator,
            limb,
            end_bit,
            start_bit,
        };
        self.register_instruction(instruction);

        bit
    }
}

impl<AP: AirParser> AirConstraint<AP> for LimbBitInstruction {
    fn eval(&self, parser: &mut AP) {
        // Assert the initial valuen of `bit_accumulator` at the begining of each cycle. As the bits
        // are presented in little-endian order, the initial value of `bit_accumulator` is the value
        // of the limb register at the beginning of the cycle. This translates to the constraint:
        //    `start_bit * (bit_accumulator - limb) = 0`
        let bit_accumulator = self.bit_accumulator.eval(parser);
        let start_bit = self.start_bit.eval(parser);
        let limb_register = self.limb.eval(parser);
        let mut limb_constraint = parser.sub(bit_accumulator, limb_register);
        limb_constraint = parser.mul(start_bit, limb_constraint);
        parser.constraint(limb_constraint);

        // Assert that the bit accumulator is summed correctly. For that purpose, we assert that for
        // every row other than the last one in the cycle (determined by end_bit) we have that
        //     `bit_accumulator_next = (bit_accumulator - bit) / 2.`
        // And, that in the end of the cycle, we have:
        //     `bit_accumulator = bit`
        //
        // This translates to the constraints:
        //     `end_bit.not() * (2 * bit_accumulator_next - bit_accumulator + bit) = 0`
        //     `end_bit * (bit_accumulator - bit) = 0`
        let bit = self.bit.eval(parser);
        let end_bit = self.end_bit.eval(parser);
        let one = parser.one();
        let not_end_bit = parser.sub(one, end_bit);

        // Constrain `end_bit.not() * (2 * bit_accumulator_next - bit_accumulator + bit) = 0`
        let mut transition_constraint = self.bit_accumulator.next().eval(parser);
        transition_constraint =
            parser.mul_const(transition_constraint, AP::Field::from_canonical_u8(2));
        transition_constraint = parser.sub(transition_constraint, bit_accumulator);
        transition_constraint = parser.add(transition_constraint, bit);
        transition_constraint = parser.mul(not_end_bit, transition_constraint);
        parser.constraint_transition(transition_constraint);

        // Constrain `end_bit * (bit_accumulator - bit) = 0`
        let mut end_constraint = parser.sub(bit_accumulator, bit);
        end_constraint = parser.mul(end_bit, end_constraint);
        parser.constraint(end_constraint);
    }
}

impl<F: PrimeField64> Instruction<F> for LimbBitInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        // Load the limb value and write the correct bit.
        let limb = writer.read(&self.limb, row_index);
        let limb_u32 = limb.as_canonical_u64() as u32;

        let bit_index = row_index % 32;
        let bit = (limb_u32 >> bit_index) & 1;
        writer.write(&self.bit, &F::from_canonical_u32(bit), row_index);

        // Write the bit accumulator.
        let start_bit = writer.read(&self.start_bit, row_index) == F::ONE;
        let end_bit = writer.read(&self.end_bit, row_index) == F::ONE;

        // If this is the first bit, then the bit accumulator is the limb value.
        if start_bit {
            writer.write(&self.bit_accumulator, &limb, row_index);
        }

        // Unless this is the last bit, the next bit accumulator is given as:
        // `bit_accumulator_next = (bit_accumulator - bit) / 2.`
        if !end_bit {
            let bit_accumulator = writer
                .read(&self.bit_accumulator, row_index)
                .as_canonical_u64() as u32;
            let next_value = F::from_canonical_u32((bit_accumulator - bit) / 2);
            writer.write(&self.bit_accumulator.next(), &next_value, row_index);
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        // Load the limb value and write the correct bit.
        let limb = writer.read(&self.limb);
        let limb_u32 = limb.as_canonical_u64() as u32;

        let bit_index = writer.row_index().unwrap() % 32;
        let bit = (limb_u32 >> bit_index) & 1;
        writer.write(&self.bit, &F::from_canonical_u32(bit));

        // Write the bit accumulator.
        let start_bit = writer.read(&self.start_bit) == F::ONE;
        let end_bit = writer.read(&self.end_bit) == F::ONE;

        // If this is the first bit, then the bit accumulator is the limb value.
        if start_bit {
            writer.write(&self.bit_accumulator, &limb);
        }

        // Unless this is the last bit, the next bit accumulator is given as:
        // `bit_accumulator_next = (bit_accumulator - bit) / 2.`
        if !end_bit {
            let bit_accumulator = writer.read(&self.bit_accumulator).as_canonical_u64() as u32;
            let next_value = F::from_canonical_u32((bit_accumulator - bit) / 2);
            writer.write(&self.bit_accumulator.next(), &next_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use rand::Rng;

    use super::*;
    use crate::chip::builder::tests::ArithmeticGenerator;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    use crate::plonky2::stark::tests::{test_recursive_starky, test_starky};
    use crate::plonky2::stark::Starky;
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct BitDecompTest;

    impl AirParameters for BitDecompTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = LimbBitInstruction;

        const NUM_FREE_COLUMNS: usize = 8;
    }

    #[test]
    fn test_bit_decomposition_instruction() {
        type F = GoldilocksField;
        type L = BitDecompTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let limb = builder.alloc::<ElementRegister>();
        let cycle_32 = builder.cycle(5);

        let bit = builder.bit_decomposition(limb, cycle_32.start_bit, cycle_32.end_bit);

        let num_rows = 1 << 6;

        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let writer = generator.new_writer();

        let mut rng = rand::thread_rng();
        let limbs = (0..(num_rows / 32))
            .map(|_| rng.gen())
            .collect::<Vec<u32>>();
        for i in 0..num_rows {
            let limb_index = i / 32;
            writer.write(&limb, &F::from_canonical_u32(limbs[limb_index]), i);
            writer.write_row_instructions(&generator.air_data, i);
        }

        for (limb, row_index) in limbs.iter().zip((0..num_rows).step_by(32)) {
            let value_from_bits = (0..32)
                .map(|i| {
                    let bit = writer.read(&bit, row_index + i).as_canonical_u64() as u32;
                    bit << i
                })
                .sum::<u32>();
            assert_eq!(value_from_bits, *limb);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        let public_inputs = writer.public.read().unwrap().clone();
        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
