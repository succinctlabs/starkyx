use core::str::FromStr;

use num::BigUint;
use serde::{Deserialize, Serialize};

use super::params::Ed25519BaseField;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::field::register::FieldRegister;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::u16::U16Register;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::utils::digits_to_biguint;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::to_u16_le_limbs_polynomial;

/// Fp Square Root. Computes `sqrt(a) = result`.
///
/// This is done by witnessing the square root and then constraining that result * result == a.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Ed25519FpSqrtInstruction {
    /// a `FpMulInstruction` to compute `a * a = result`.
    square: FpMulInstruction<Ed25519BaseField>,
    /// Witness the bits of the least significant limb (skipping the first bit).
    limb_witness: ArrayRegister<BitRegister>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// given two field elements `a` and `b`, computes a positive square root.
    ///
    /// WARNING: While trace generation will give the correct result which is whithin the rangwe of
    /// the field modulus, there are no constraints checking that and such checks must be done by
    /// the caller.
    pub fn ed25519_sqrt(
        &mut self,
        a: &FieldRegister<Ed25519BaseField>,
    ) -> FieldRegister<Ed25519BaseField>
    where
        L::Instruction: From<Ed25519FpSqrtInstruction>,
    {
        let is_trace = a.is_trace();
        let result = if is_trace {
            self.alloc::<FieldRegister<Ed25519BaseField>>()
        } else {
            self.alloc_public::<FieldRegister<Ed25519BaseField>>()
        };
        self.set_ed25519_sqrt(a, &result);
        result
    }

    pub fn set_ed25519_sqrt(
        &mut self,
        a: &FieldRegister<Ed25519BaseField>,
        result: &FieldRegister<Ed25519BaseField>,
    ) where
        L::Instruction: From<Ed25519FpSqrtInstruction>,
    {
        let is_trace = a.is_trace() || result.is_trace();

        let square_carry: FieldRegister<Ed25519BaseField>;
        let square_witness_low: ArrayRegister<U16Register>;
        let square_witness_high: ArrayRegister<U16Register>;
        let limb_witness: ArrayRegister<BitRegister>;

        if is_trace {
            square_carry = self.alloc::<FieldRegister<Ed25519BaseField>>();
            square_witness_low =
                self.alloc_array::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
            square_witness_high =
                self.alloc_array::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
            limb_witness = self.alloc_array::<BitRegister>(Ed25519BaseField::NB_BITS_PER_LIMB - 1);
        } else {
            square_carry = self.alloc_public::<FieldRegister<Ed25519BaseField>>();
            square_witness_low =
                self.alloc_array_public::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
            square_witness_high =
                self.alloc_array_public::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
            limb_witness =
                self.alloc_array_public::<BitRegister>(Ed25519BaseField::NB_BITS_PER_LIMB - 1);
        }

        // check that a_sqrt * a_sqrt == a
        let square = FpMulInstruction {
            a: *result,
            b: *result,
            result: *a,
            carry: square_carry,
            witness_low: square_witness_low,
            witness_high: square_witness_high,
        };

        let instr = Ed25519FpSqrtInstruction {
            square,
            limb_witness,
        };

        if is_trace {
            self.register_instruction(instr);
        } else {
            self.register_global_instruction(instr);
        }
    }
}

impl<AP: PolynomialParser> AirConstraint<AP> for Ed25519FpSqrtInstruction {
    fn eval(&self, parser: &mut AP) {
        // Assert that result * result == a
        self.square.eval(parser);

        // Assert that the least significant bit of the square root is zero, by witnessing all other
        // bits of the least significant limb.
        let mut acc = parser.zero();
        for (i, bit) in self.limb_witness.iter().enumerate() {
            let bit = bit.eval(parser);
            let two_i = parser.constant(AP::Field::from_canonical_u32(1 << (i + 1)));
            let bit_two_i = parser.mul(two_i, bit);
            acc = parser.add(acc, bit_two_i);
        }
        let limb = self.square.a.eval(parser).coefficients[0];
        parser.assert_eq(limb, acc);
    }
}

impl<F: PrimeField64> Instruction<F> for Ed25519FpSqrtInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let p_a = writer.read(&self.square.result, row_index);

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let a = digits_to_biguint(&a_digits);

        let beta = sqrt(a);
        let p_beta = to_u16_le_limbs_polynomial::<F, Ed25519BaseField>(&beta);
        let a = &self.square.a;

        let limb = p_beta.coefficients[0].as_canonical_u64();
        let limb_bits = (0..Ed25519BaseField::NB_BITS_PER_LIMB)
            .map(|i| F::from_canonical_u64((limb >> i) & 1))
            .skip(1);

        writer.write(a, &p_beta, row_index);
        writer.write_array(&self.limb_witness, limb_bits, row_index);

        self.square.write(writer, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let p_a = writer.read(&self.square.result);

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let a = digits_to_biguint(&a_digits);

        let beta = sqrt(a);
        let p_beta = to_u16_le_limbs_polynomial::<F, Ed25519BaseField>(&beta);
        let a = &self.square.a;

        let limb = p_beta.coefficients[0].as_canonical_u64();
        let limb_bits = (0..Ed25519BaseField::NB_BITS_PER_LIMB)
            .map(|i| F::from_canonical_u64((limb >> i) & 1))
            .skip(1);

        writer.write(a, &p_beta);
        writer.write_array(&self.limb_witness, limb_bits);

        self.square.write_to_air(writer);
    }
}

pub fn sqrt(a: BigUint) -> BigUint {
    // Here is a description of how to calculate sqrt in the Curve25519 base field:
    // https://github.com/succinctlabs/curve25519-dalek/blob/e2d1bd10d6d772af07cac5c8161cd7655016af6d/curve25519-dalek/src/field.rs#L256

    let modulus = Ed25519BaseField::modulus();
    // The exponent is (modulus+3)/8;
    let mut beta = a.modpow(
        &BigUint::from_str(
            "7237005577332262213973186563042994240829374041602535252466099000494570602494",
        )
        .unwrap(),
        &modulus,
    );

    // The square root of -1 in the field.
    // Take from here:
    // https://github.com/succinctlabs/curve25519-dalek/blob/e2d1bd10d6d772af07cac5c8161cd7655016af6d/curve25519-dalek/src/backend/serial/u64/constants.rs#L89
    let sqrt_m1 = BigUint::from_str(
        "19681161376707505956807079304988542015446066515923890162744021073123829784752",
    )
    .unwrap();

    let beta_squared = &beta * &beta % &modulus;
    let neg_a = &modulus - &a;

    if beta_squared == neg_a {
        beta = (&beta * &sqrt_m1) % &modulus;
    }

    let correct_sign_sqrt = beta_squared == a;
    let flipped_sign_sqrt = beta_squared == neg_a;

    if !correct_sign_sqrt && !flipped_sign_sqrt {
        panic!("a is not a square");
    }

    let beta_bytes = beta.to_bytes_le();
    if (beta_bytes[0] & 1) == 1 {
        beta = (&modulus - &beta) % &modulus;
    }

    beta
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::polynomial::Polynomial;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    struct FpSqrtTest;

    impl AirParameters for FpSqrtTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 108;
        const NUM_FREE_COLUMNS: usize = 17;
        const EXTENDED_COLUMNS: usize = 171;

        type Instruction = Ed25519FpSqrtInstruction;
    }

    #[test]
    fn test_fpsqrt() {
        type F = GoldilocksField;
        type L = FpSqrtTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Ed25519BaseField;

        let p = Ed25519BaseField::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a_pub = builder.alloc_public::<FieldRegister<P>>();
        let result_pub = builder.ed25519_sqrt(&a_pub);

        let a = builder.alloc::<FieldRegister<P>>();
        let result = builder.ed25519_sqrt(&a);

        let (air, trace_data) = builder.build();
        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let writer = generator.new_writer();
            let a_sqrt_int = rng.gen_biguint(256) % &p;
            let a_int = (&a_sqrt_int * &a_sqrt_int) % &p;
            let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
            let p_a_sqrt = Polynomial::<F>::from_biguint_field(&a_sqrt_int, 16, 16);

            writer.write(&a, &p_a, i);
            writer.write(&result, &p_a_sqrt, i);
            writer.write(&a_pub, &p_a, i);
            writer.write(&result_pub, &p_a_sqrt, i);
            writer.write_row_instructions(&generator.air_data, i);
        }

        let writer = generator.new_writer();
        writer.write_global_instructions(&generator.air_data);

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);
        let public = writer.public().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }
}
