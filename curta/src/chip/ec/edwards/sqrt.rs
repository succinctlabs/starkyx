use core::str::FromStr;

use num::BigUint;
use serde::{Deserialize, Serialize};

use super::ed25519::Ed25519BaseField;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::field::register::FieldRegister;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::u16::U16Register;
use crate::chip::register::RegisterSerializable;
use crate::chip::trace::writer::TraceWriter;
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
pub struct FpSqrtInstruction {
    /// a `FpMulInstruction` to compute `a * a = result`.
    square: FpMulInstruction<Ed25519BaseField>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// given two field elements `a` and `b`, computes the quotient `a / b = c`.
    pub fn fp_sqrt(
        &mut self,
        a: &FieldRegister<Ed25519BaseField>,
    ) -> FieldRegister<Ed25519BaseField>
    where
        L::Instruction: From<FpSqrtInstruction>,
    {
        let is_trace = a.is_trace();
        let result = if is_trace {
            self.alloc::<FieldRegister<Ed25519BaseField>>()
        } else {
            self.alloc_public::<FieldRegister<Ed25519BaseField>>()
        };
        self.set_fp_sqrt(a, &result);
        result
    }

    pub fn set_fp_sqrt(
        &mut self,
        a: &FieldRegister<Ed25519BaseField>,
        result: &FieldRegister<Ed25519BaseField>,
    ) where
        L::Instruction: From<FpSqrtInstruction>,
    {
        let is_trace = a.is_trace() || result.is_trace();

        let square_carry: FieldRegister<Ed25519BaseField>;
        let square_witness_low: ArrayRegister<U16Register>;
        let square_witness_high: ArrayRegister<U16Register>;

        if is_trace {
            square_carry = self.alloc::<FieldRegister<Ed25519BaseField>>();
            square_witness_low =
                self.alloc_array::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
            square_witness_high =
                self.alloc_array::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
        } else {
            square_carry = self.alloc_public::<FieldRegister<Ed25519BaseField>>();
            square_witness_low =
                self.alloc_array_public::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
            square_witness_high =
                self.alloc_array_public::<U16Register>(Ed25519BaseField::NB_WITNESS_LIMBS);
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

        let instr = FpSqrtInstruction { square };

        if is_trace {
            self.register_instruction(instr);
        } else {
            self.register_global_instruction(instr);
        }
    }
}

impl<AP: PolynomialParser> AirConstraint<AP> for FpSqrtInstruction {
    fn eval(&self, parser: &mut AP) {
        self.square.eval(parser);
    }
}

impl<F: PrimeField64> Instruction<F> for FpSqrtInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let p_a = writer.read(&self.square.result, row_index);

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        // Here is a description of how to calculate sqrt in the Curve25519 base field:
        // https://github.com/succinctlabs/curve25519-dalek/blob/main/curve25519-dalek/src/field.rs#L256
        let a = digits_to_biguint(&a_digits);

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
        let sqrt_m1 = BigUint::from_str(
            "19681161376707505956807079304988542015446066515923890162744021073123829784752",
        )
        .unwrap();

        let beta_squared = &beta * &beta % &modulus;
        let neg_a = &modulus - &a;

        if beta_squared == neg_a {
            beta = &beta * &sqrt_m1;
        }

        let correct_sign_sqrt = beta_squared == a;
        let flipped_sign_sqrt = beta_squared == neg_a;

        if !correct_sign_sqrt && !flipped_sign_sqrt {
            panic!("a is not a square");
        }

        if flipped_sign_sqrt {
            beta = (&beta * &sqrt_m1) % &modulus;
        }

        let p_beta = to_u16_le_limbs_polynomial::<F, Ed25519BaseField>(&beta);

        let a = &self.square.a;
        writer.write(a, &p_beta, row_index);

        self.square.write(writer, row_index);
    }
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
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 171;

        type Instruction = FpSqrtInstruction;
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
        let result_pub = builder.fp_sqrt(&a_pub);

        let a = builder.alloc::<FieldRegister<P>>();
        let result = builder.fp_sqrt(&a);

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
