//! Implements non-native inner product as an "instruction".
//!
//! To understand the implementation, it may be useful to refer to `mod.rs`.

use num::{BigUint, Zero};
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
use crate::chip::utils::{field_limbs_to_biguint, split_u32_limbs_to_u16_limbs};
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::{to_u16_le_limbs_polynomial, Polynomial};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FpInnerProductInstruction<P: FieldParameters> {
    a: Vec<FieldRegister<P>>,
    b: Vec<FieldRegister<P>>,
    pub result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn fp_inner_product<P: FieldParameters>(
        &mut self,
        a: &[FieldRegister<P>],
        b: &[FieldRegister<P>],
    ) -> FieldRegister<P>
    where
        L::Instruction: From<FpInnerProductInstruction<P>>,
    {
        debug_assert!(a.len() == b.len());

        let is_trace = a.iter().any(|x| x.is_trace()) || b.iter().any(|x| x.is_trace());

        let result: FieldRegister<P>;
        let carry: FieldRegister<P>;
        let witness_low: ArrayRegister<U16Register>;
        let witness_high: ArrayRegister<U16Register>;
        if is_trace {
            result = self.alloc::<FieldRegister<P>>();
            carry = self.alloc::<FieldRegister<P>>();
            witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
            witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        } else {
            result = self.alloc_public::<FieldRegister<P>>();
            carry = self.alloc_public::<FieldRegister<P>>();
            witness_low = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
            witness_high = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
        }

        let instr = FpInnerProductInstruction {
            a: a.to_vec(),
            b: b.to_vec(),
            result,
            carry,
            witness_low,
            witness_high,
        };

        if is_trace {
            self.register_instruction(instr.clone());
        } else {
            self.register_global_instruction(instr.clone());
        }
        result
    }
}

impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpInnerProductInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        let p_a_vec = self.a.iter().map(|x| x.eval(parser)).collect::<Vec<_>>();
        let p_b_vec = self.b.iter().map(|x| x.eval(parser)).collect::<Vec<_>>();

        let p_result = self.result.eval(parser);
        let p_carry = self.carry.eval(parser);

        let p_zero = parser.zero_poly();

        let p_inner_product = p_a_vec
            .iter()
            .zip(p_b_vec.iter())
            .map(|(p_a, p_b)| parser.poly_mul(p_a, p_b))
            .collect::<Vec<_>>()
            .iter()
            .fold(p_zero, |acc, x| parser.poly_add(&acc, x));

        let p_inner_product_minus_result = parser.poly_sub(&p_inner_product, &p_result);
        let p_limbs = parser.constant_poly(&Polynomial::from_iter(util::modulus_field_iter::<
            AP::Field,
            P,
        >()));
        let p_carry_mul_modulus = parser.poly_mul(&p_carry, &p_limbs);
        let p_vanishing = parser.poly_sub(&p_inner_product_minus_result, &p_carry_mul_modulus);

        let p_witness_low = Polynomial::from_coefficients(self.witness_low.eval_vec(parser));
        let p_witness_high = Polynomial::from_coefficients(self.witness_high.eval_vec(parser));

        util::eval_field_operation::<AP, P>(parser, &p_vanishing, &p_witness_low, &p_witness_high)
    }
}

impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpInnerProductInstruction<P> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        // Make little endian polynomial limbs.
        let p_a_vec = self
            .a
            .iter()
            .map(|a| writer.read(a, row_index))
            .collect::<Vec<Polynomial<F>>>();
        let p_b_vec = self
            .b
            .iter()
            .map(|b| writer.read(b, row_index))
            .collect::<Vec<Polynomial<F>>>();

        let modulus = &P::modulus();
        let inner_product = p_a_vec
            .iter()
            .zip(p_b_vec.iter())
            .map(|(a, b)| {
                (
                    field_limbs_to_biguint(a.coefficients()),
                    field_limbs_to_biguint(b.coefficients()),
                )
            })
            .fold(BigUint::zero(), |acc, (c, d)| acc + c * d);

        let result = &(&inner_product % modulus);
        let carry = &((&inner_product - result) / modulus);
        assert!(result < modulus);
        assert!(carry < &(2u32 * modulus));
        assert_eq!(carry * modulus, inner_product - result);

        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(carry);

        // Compute the vanishing polynomial.
        let p_inner_product = p_a_vec.into_iter().zip(p_b_vec).fold(
            Polynomial::<F>::from_coefficients(vec![F::ZERO]),
            |acc, (c, d)| acc + &c * &d,
        );
        let p_vanishing = p_inner_product - &p_result - &p_carry * &p_modulus;
        assert_eq!(p_vanishing.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness
        let p_witness = util::compute_root_quotient_and_shift(&p_vanishing, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs(&p_witness);

        writer.write(&self.result, &p_result, row_index);
        writer.write(&self.carry, &p_carry, row_index);
        writer.write_array(&self.witness_low, &p_witness_low, row_index);
        writer.write_array(&self.witness_high, &p_witness_high, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        // Make little endian polynomial limbs.
        let p_a_vec = self
            .a
            .iter()
            .map(|a| writer.read(a))
            .collect::<Vec<Polynomial<F>>>();
        let p_b_vec = self
            .b
            .iter()
            .map(|b| writer.read(b))
            .collect::<Vec<Polynomial<F>>>();

        let modulus = &P::modulus();
        let inner_product = p_a_vec
            .iter()
            .zip(p_b_vec.iter())
            .map(|(a, b)| {
                (
                    field_limbs_to_biguint(a.coefficients()),
                    field_limbs_to_biguint(b.coefficients()),
                )
            })
            .fold(BigUint::zero(), |acc, (c, d)| acc + c * d);

        let result = &(&inner_product % modulus);
        let carry = &((&inner_product - result) / modulus);
        assert!(result < modulus);
        assert!(carry < &(2u32 * modulus));
        assert_eq!(carry * modulus, inner_product - result);

        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(carry);

        // Compute the vanishing polynomial.
        let p_inner_product = p_a_vec.into_iter().zip(p_b_vec).fold(
            Polynomial::<F>::from_coefficients(vec![F::ZERO]),
            |acc, (c, d)| acc + &c * &d,
        );
        let p_vanishing = p_inner_product - &p_result - &p_carry * &p_modulus;
        assert_eq!(p_vanishing.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness
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
    use rand::thread_rng;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::field::parameters::tests::Fp25519;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    struct FpInnerProductTest;

    impl AirParameters for FpInnerProductTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 156;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 243;

        type Instruction = FpInnerProductInstruction<Fp25519>;
    }

    #[test]
    fn test_fpquad() {
        type F = GoldilocksField;
        type L = FpInnerProductTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;
        type Fp = FieldRegister<P>;

        let p = Fp25519::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a_pub = builder.alloc_public::<Fp>();
        let b_pub = builder.alloc_public::<Fp>();
        let c_pub = builder.alloc_public::<Fp>();
        let d_pub = builder.alloc_public::<Fp>();
        let _ = builder.fp_inner_product(&[a_pub, b_pub], &[c_pub, d_pub]);

        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();
        let c = builder.alloc::<Fp>();
        let d = builder.alloc::<Fp>();
        let _ = builder.fp_inner_product(&[a, b], &[c, d]);

        let (air, trace_data) = builder.build();
        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let writer = generator.new_writer();
            let a_int = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let c_int = rng.gen_biguint(256) % &p;
            let d_int = rng.gen_biguint(256) % &p;
            let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
            let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);
            let p_c = Polynomial::<F>::from_biguint_field(&c_int, 16, 16);
            let p_d = Polynomial::<F>::from_biguint_field(&d_int, 16, 16);

            writer.write(&a, &p_a, i);
            writer.write(&b, &p_b, i);
            writer.write(&c, &p_c, i);
            writer.write(&d, &p_d, i);

            writer.write(&a_pub, &p_a, i);
            writer.write(&b_pub, &p_b, i);
            writer.write(&c_pub, &p_c, i);
            writer.write(&d_pub, &p_d, i);

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
