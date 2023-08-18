use num::BigUint;

use super::parameters::FieldParameters;
use super::register::FieldRegister;
use super::util;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::u16::U16Register;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::utils::{
    compute_root_quotient_and_shift, field_limbs_to_biguint, split_u32_limbs_to_u16_limbs,
};
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::{to_u16_le_limbs_polynomial, Polynomial};

#[derive(Debug, Clone, Copy)]
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
    pub fn fp_den<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
        sign: bool,
    ) -> FpDenInstruction<P>
    where
        L::Instruction: From<FpDenInstruction<P>>,
    {
        let result = self.alloc::<FieldRegister<P>>();
        let carry = self.alloc::<FieldRegister<P>>();
        let witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let instr = FpDenInstruction {
            a: *a,
            b: *b,
            sign,
            result,
            carry,
            witness_low,
            witness_high,
        };
        self.register_instruction(instr);
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
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.result.register(),
            *self.carry.register(),
            *self.witness_low.register(),
            *self.witness_high.register(),
        ]
    }

    fn inputs(&self) -> Vec<MemorySlice> {
        vec![*self.a.register(), *self.b.register()]
    }

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

        // Row must match layout of instruction.
        let mut values = p_result.coefficients;
        values.extend_from_slice(p_carry.coefficients());
        values.extend_from_slice(&p_witness_low);
        values.extend_from_slice(&p_witness_high);

        // Row must match layout of instruction.
        writer.write_unsafe_batch_raw(
            &[
                *self.result.register(),
                *self.carry.register(),
                *self.witness_low.register(),
                *self.witness_high.register(),
            ],
            &values,
            row_index,
        )
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

    #[derive(Clone, Debug, Copy)]
    struct DenTest;

    impl const AirParameters for DenTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 140;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 219;

        type Instruction = FpDenInstruction<Fp25519>;

        fn num_rows_bits() -> usize {
            16
        }
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

        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();
        let sign = false;
        let den_ins = builder.fp_den(&a, &b, sign);

        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data);

        let (tx, rx) = channel();
        let mut rng = thread_rng();
        for i in 0..L::num_rows() {
            let writer = generator.new_writer();
            let handle = tx.clone();
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            rayon::spawn(move || {
                let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
                let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);

                writer.write(&a, &p_a, i);
                writer.write(&b, &p_b, i);
                writer.write_instruction(&den_ins, i);

                handle.send(1).unwrap();
            });
        }
        drop(tx);
        for msg in rx.iter() {
            assert!(msg == 1);
        }
        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
