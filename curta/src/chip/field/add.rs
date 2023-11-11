use serde::{Deserialize, Serialize};

use super::parameters::FieldParameters;
use super::register::FieldRegister;
use super::util;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::u16::U16Register;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::utils::{digits_to_biguint, split_u32_limbs_to_u16_limbs};
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::{to_u16_le_limbs_polynomial, Polynomial};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FpAddInstruction<P: FieldParameters> {
    pub a: FieldRegister<P>,
    pub b: FieldRegister<P>,
    pub result: FieldRegister<P>,
    pub(crate) carry: FieldRegister<P>,
    pub(crate) witness_low: ArrayRegister<U16Register>,
    pub(crate) witness_high: ArrayRegister<U16Register>,
}

impl<P: FieldParameters> FpAddInstruction<P> {
    pub fn set_inputs(&mut self, a: &FieldRegister<P>, b: &FieldRegister<P>) {
        self.a = *a;
        self.b = *b;
    }
}

impl<L: AirParameters> AirBuilder<L> {
    /// Given two field elements `a` and `b`, computes the sum `a + b = c`.
    pub fn fp_add<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
    ) -> FieldRegister<P>
    where
        L::Instruction: From<FpAddInstruction<P>>,
    {
        let is_trace = a.is_trace() || b.is_trace();
        let result = if is_trace {
            self.alloc::<FieldRegister<P>>()
        } else {
            self.alloc_public::<FieldRegister<P>>()
        };
        self.set_fp_add(a, b, &result);

        result
    }

    pub fn set_fp_add<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
        result: &FieldRegister<P>,
    ) where
        L::Instruction: From<FpAddInstruction<P>>,
    {
        let is_trace = a.is_trace() || b.is_trace() || result.is_trace();
        let carry: FieldRegister<P>;
        let witness_low: ArrayRegister<U16Register>;
        let witness_high: ArrayRegister<U16Register>;
        if is_trace {
            carry = self.alloc::<FieldRegister<P>>();
            witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
            witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        } else {
            carry = self.alloc_public::<FieldRegister<P>>();
            witness_low = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
            witness_high = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
        }
        let instr = FpAddInstruction {
            a: *a,
            b: *b,
            result: *result,
            carry,
            witness_low,
            witness_high,
        };
        if is_trace {
            self.register_instruction(instr);
        } else {
            self.register_global_instruction(instr);
        }
    }
}

// Constraints for FpAddInstruction
impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpAddInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        let p_a = self.a.eval(parser);
        let p_b = self.b.eval(parser);
        let p_result = self.result.eval(parser);
        let p_carry = self.carry.eval(parser);

        let p_a_plus_b = parser.poly_add(&p_a, &p_b);
        let p_a_plus_b_minus_result = parser.poly_sub(&p_a_plus_b, &p_result);
        let p_limbs = parser.constant_poly(&Polynomial::from_iter(util::modulus_field_iter::<
            AP::Field,
            P,
        >()));

        let p_mul_times_carry = parser.poly_mul(&p_carry, &p_limbs);
        let p_vanishing = parser.poly_sub(&p_a_plus_b_minus_result, &p_mul_times_carry);

        let p_witness_low = Polynomial::from_coefficients(self.witness_low.eval_vec(parser));
        let p_witness_high = Polynomial::from_coefficients(self.witness_high.eval_vec(parser));

        util::eval_field_operation::<AP, P>(parser, &p_vanishing, &p_witness_low, &p_witness_high)
    }
}

// Instruction trait
impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpAddInstruction<P> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let p_a = writer.read(&self.a, row_index);
        let p_b = writer.read(&self.b, row_index);

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();
        let b_digits = p_b
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let a = digits_to_biguint(&a_digits);
        let b = digits_to_biguint(&b_digits);

        // Compute field addition in the integers.
        let modulus = P::modulus();
        let result = (&a + &b) % &modulus;
        let carry = (&a + &b - &result) / &modulus;
        debug_assert!(result < modulus);
        debug_assert!(carry < modulus);
        debug_assert_eq!(&carry * &modulus, a + b - &result);

        // Make little endian polynomial limbs.
        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(&modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let p_vanishing = &p_a + &p_b - &p_result - &p_carry * &p_modulus;
        debug_assert_eq!(p_vanishing.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness = util::compute_root_quotient_and_shift(&p_vanishing, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs(&p_witness);

        writer.write(&self.result, &p_result, row_index);
        writer.write(&self.carry, &p_carry, row_index);
        writer.write_array(&self.witness_low, &p_witness_low, row_index);
        writer.write_array(&self.witness_high, &p_witness_high, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let p_a = writer.read(&self.a);
        let p_b = writer.read(&self.b);

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();
        let b_digits = p_b
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let a = digits_to_biguint(&a_digits);
        let b = digits_to_biguint(&b_digits);

        // Compute field addition in the integers.
        let modulus = P::modulus();
        let result = (&a + &b) % &modulus;
        let carry = (&a + &b - &result) / &modulus;
        debug_assert!(result < modulus);
        debug_assert!(carry < modulus);
        debug_assert_eq!(&carry * &modulus, a + b - &result);

        // Make little endian polynomial limbs.
        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(&modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let p_vanishing = &p_a + &p_b - &p_result - &p_carry * &p_modulus;
        debug_assert_eq!(p_vanishing.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness = util::compute_root_quotient_and_shift(&p_vanishing, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs(&p_witness);

        writer.write(&self.result, &p_result);
        writer.write(&self.carry, &p_carry);
        writer.write_array(&self.witness_low, &p_witness_low);
        writer.write_array(&self.witness_high, &p_witness_high);
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use num::BigUint;
    use rand::thread_rng;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::field::parameters::tests::Fp25519;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    struct FpAddTest;

    impl AirParameters for FpAddTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 195;

        type Instruction = FpAddInstruction<Fp25519>;
    }

    #[test]
    fn test_fpadd() {
        type F = GoldilocksField;
        type L = FpAddTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;

        let p = Fp25519::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a_pub = builder.alloc_public::<FieldRegister<P>>();
        let b_pub = builder.alloc_public::<FieldRegister<P>>();
        let _ = builder.fp_add(&a_pub, &b_pub);

        let a = builder.alloc::<FieldRegister<P>>();
        let b = builder.alloc::<FieldRegister<P>>();
        let _add_insr = builder.fp_add(&a, &b);

        let (air, trace_data) = builder.build();
        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let trace_initial = (0..num_rows)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let writer = generator.new_writer();
                // let handle = tx.clone();
                let a_int: BigUint = rng.gen_biguint(256) % &p;
                let b_int = rng.gen_biguint(256) % &p;
                (writer, a_int, b_int)
            })
            .collect::<Vec<_>>();

        trace_initial
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (writer, a_int, b_int))| {
                let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
                let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);

                writer.write_slice(&a, p_a.coefficients(), i);
                writer.write_slice(&b, p_b.coefficients(), i);

                writer.write_slice(&a_pub, p_a.coefficients(), i);
                writer.write_slice(&b_pub, p_b.coefficients(), i);

                writer.write_row_instructions(&generator.air_data, i);
            });

        let writer = generator.new_writer();
        writer.write_global_instructions(&generator.air_data);

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        let writer = generator.new_writer();
        let public = writer.public().unwrap().clone();
        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }
}
