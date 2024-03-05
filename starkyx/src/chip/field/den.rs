use num::BigUint;
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
use crate::chip::utils::{
    compute_root_quotient_and_shift, field_limbs_to_biguint, split_u32_limbs_to_u16_limbs,
};
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::{to_u16_le_limbs_polynomial, Polynomial};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FpDenInstruction<P: FieldParameters> {
    a: FieldRegister<P>,
    b: FieldRegister<P>,
    sign: bool,
    pub result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// Computes the element `a / (1 - b)` when `sign == 0` and `a / (1 + b)` when `sign == 1`.
    ///
    /// The constraints in `fp_den` only check that `result * denominator = a mod p`, they do NOT
    /// check that the denominator is non-zero[^1].
    ///
    /// [^1]: The reason is that the main use of this instruction is for Edwards addition in which we
    ///  know that the denominator is non-zero.
    pub fn fp_den<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
        sign: bool,
    ) -> FpDenInstruction<P>
    where
        L::Instruction: From<FpDenInstruction<P>>,
    {
        let is_trace = a.is_trace() || b.is_trace();

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
        let instr = FpDenInstruction {
            a: *a,
            b: *b,
            sign,
            result,
            carry,
            witness_low,
            witness_high,
        };
        if is_trace {
            self.register_instruction(instr);
        } else {
            self.register_global_instruction(instr);
        }
        instr
    }
}

impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpDenInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        let p_a = self.a.eval(parser);
        let p_b = self.b.eval(parser);
        let p_result = self.result.eval(parser);
        let p_carry = self.carry.eval(parser);

        // Compute the vanishing polynomial:
        //      lhs(x) = sign * (b(x) * result(x) + result(x)) + (1 - sign) * (b(x) * result(x) + a(x))
        //      rhs(x) = sign * a(x) + (1 - sign) * result(x)
        //      lhs(x) - rhs(x) - carry(x) * p(x)
        let p_b_times_res = parser.poly_mul(&p_b, &p_result);
        let p_equation_lhs = if self.sign {
            parser.poly_add(&p_b_times_res, &p_result)
        } else {
            parser.poly_add(&p_b_times_res, &p_a)
        };
        let p_equation_rhs = if self.sign { p_a } else { p_result };

        let p_lhs_minus_rhs = parser.poly_sub(&p_equation_lhs, &p_equation_rhs);
        let p_limbs = parser.constant_poly(&Polynomial::from_iter(util::modulus_field_iter::<
            AP::Field,
            P,
        >()));

        let mul_times_carry = parser.poly_mul(&p_carry, &p_limbs);
        let p_vanishing = parser.poly_sub(&p_lhs_minus_rhs, &mul_times_carry);

        let p_witness_low = Polynomial::from_coefficients(self.witness_low.eval_vec(parser));
        let p_witness_high = Polynomial::from_coefficients(self.witness_high.eval_vec(parser));

        util::eval_field_operation::<AP, P>(parser, &p_vanishing, &p_witness_low, &p_witness_high)
    }
}

impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpDenInstruction<P> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let p_a = writer.read(&self.a, row_index);
        let p_b = writer.read(&self.b, row_index);

        let a = field_limbs_to_biguint(p_a.coefficients());
        let b = field_limbs_to_biguint(p_b.coefficients());

        let p = P::modulus();
        let minus_b_int = &p - &b;
        let b_signed = if self.sign { &b } else { &minus_b_int };
        let denominator = (b_signed + 1u32) % &p;
        let den_inv = denominator.modpow(&(&p - 2u32), &p);
        let result = (&a * &den_inv) % &p;
        debug_assert_eq!(&den_inv * &denominator % &p, BigUint::from(1u32));
        debug_assert!(result < p);

        let equation_lhs = if self.sign {
            &b * &result + &result
        } else {
            &b * &result + &a
        };
        let equation_rhs = if self.sign { a.clone() } else { result.clone() };
        let carry = (&equation_lhs - &equation_rhs) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, &equation_lhs - &equation_rhs);

        // Make little endian polynomial limbs.
        let p_a = to_u16_le_limbs_polynomial::<F, P>(&a);
        let p_b = to_u16_le_limbs_polynomial::<F, P>(&b);
        let p_p = to_u16_le_limbs_polynomial::<F, P>(&p);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let vanishing_poly = if self.sign {
            &p_b * &p_result + &p_result - &p_a - &p_carry * &p_p
        } else {
            &p_b * &p_result + &p_a - &p_result - &p_carry * &p_p
        };
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness_shifted = compute_root_quotient_and_shift(&vanishing_poly, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs::<F>(&p_witness_shifted);

        writer.write(&self.result, &p_result, row_index);
        writer.write(&self.carry, &p_carry, row_index);
        writer.write_array(&self.witness_low, &p_witness_low, row_index);
        writer.write_array(&self.witness_high, &p_witness_high, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let p_a = writer.read(&self.a);
        let p_b = writer.read(&self.b);

        let a = field_limbs_to_biguint(p_a.coefficients());
        let b = field_limbs_to_biguint(p_b.coefficients());

        let p = P::modulus();
        let minus_b_int = &p - &b;
        let b_signed = if self.sign { &b } else { &minus_b_int };
        let denominator = (b_signed + 1u32) % &p;
        let den_inv = denominator.modpow(&(&p - 2u32), &p);
        let result = (&a * &den_inv) % &p;
        debug_assert_eq!(&den_inv * &denominator % &p, BigUint::from(1u32));
        debug_assert!(result < p);

        let equation_lhs = if self.sign {
            &b * &result + &result
        } else {
            &b * &result + &a
        };
        let equation_rhs = if self.sign { a.clone() } else { result.clone() };
        let carry = (&equation_lhs - &equation_rhs) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, &equation_lhs - &equation_rhs);

        // Make little endian polynomial limbs.
        let p_a = to_u16_le_limbs_polynomial::<F, P>(&a);
        let p_b = to_u16_le_limbs_polynomial::<F, P>(&b);
        let p_p = to_u16_le_limbs_polynomial::<F, P>(&p);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let vanishing_poly = if self.sign {
            &p_b * &p_result + &p_result - &p_a - &p_carry * &p_p
        } else {
            &p_b * &p_result + &p_a - &p_result - &p_carry * &p_p
        };
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness_shifted = compute_root_quotient_and_shift(&vanishing_poly, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs::<F>(&p_witness_shifted);

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
    struct DenTest;

    impl AirParameters for DenTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 195;

        type Instruction = FpDenInstruction<Fp25519>;
    }

    #[test]
    fn test_fpden() {
        type F = GoldilocksField;
        type L = DenTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;
        type Fp = FieldRegister<P>;

        let p = Fp25519::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a_pub = builder.alloc_public::<Fp>();
        let b_pub = builder.alloc_public::<Fp>();
        let sign = false;
        let _ = builder.fp_den(&a_pub, &b_pub, sign);

        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();
        let sign = false;
        let _ = builder.fp_den(&a, &b, sign);

        let (air, trace_data) = builder.build();
        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let writer = generator.new_writer();
        let mut rng = thread_rng();
        let a_int: BigUint = rng.gen_biguint(256) % &p;
        let b_int = rng.gen_biguint(256) % &p;
        let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
        let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);
        writer.write(&a, &p_a, 0);
        writer.write(&b, &p_b, 0);
        writer.write_global_instructions(&generator.air_data);
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
            let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);

            writer.write(&a, &p_a, i);
            writer.write(&b, &p_b, i);

            writer.write_row_instructions(&generator.air_data, i);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);
        let public = writer.public().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }
}
