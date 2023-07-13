use std::collections::HashSet;

use num::{BigUint, Zero};

use super::parameters::{FieldParameters, MAX_NB_LIMBS};
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
use crate::chip::utils::{digits_to_biguint, split_u32_limbs_to_u16_limbs};
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::{to_u16_le_limbs_polynomial, Polynomial};

#[derive(Debug, Clone, Copy)]
pub struct FpMulConstInstruction<P: FieldParameters> {
    a: FieldRegister<P>,
    c: [u16; MAX_NB_LIMBS],
    pub result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// Given two field elements `a` and `b`, computes the product `a * b = c`.
    pub fn fp_mul_const<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        c: [u16; MAX_NB_LIMBS],
    ) -> FpMulConstInstruction<P>
    where
        L::Instruction: From<FpMulConstInstruction<P>>,
    {
        let result = self.alloc::<FieldRegister<P>>();
        let carry = self.alloc::<FieldRegister<P>>();
        let witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let instr = FpMulConstInstruction {
            a: *a,
            c,
            result,
            carry,
            witness_low,
            witness_high,
        };
        self.register_instruction(instr);
        instr
    }
}

impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpMulConstInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        let p_a = self.a.eval(parser);
        let p_c = self
            .c
            .iter()
            .map(|c| AP::Field::from_canonical_u16(*c))
            .take(P::NB_LIMBS)
            .collect::<Polynomial<AP::Field>>();
        let p_result = self.result.eval(parser);
        let p_carry = self.carry.eval(parser);

        // Compute the vanishing polynomial a(x) * b(x) - result(x) - carry(x) * p(x).
        let p_a_mul_c = parser.poly_mul_poly_const(&p_a, &p_c);
        let p_a_mul_c_minus_result = parser.poly_sub(&p_a_mul_c, &p_result);
        let p_limbs = parser.constant_poly(&Polynomial::from_iter(util::modulus_field_iter::<
            AP::Field,
            P,
        >()));

        let p_mul_times_carry = parser.poly_mul(&p_carry, &p_limbs);
        let p_vanishing = parser.poly_sub(&p_a_mul_c_minus_result, &p_mul_times_carry);

        let p_witness_low = Polynomial::from_coefficients(self.witness_low.eval_vec(parser));
        let p_witness_high = Polynomial::from_coefficients(self.witness_high.eval_vec(parser));

        util::eval_field_operation::<AP, P>(parser, &p_vanishing, &p_witness_low, &p_witness_high)
    }
}

impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpMulConstInstruction<P> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.result.register(),
            *self.carry.register(),
            *self.witness_low.register(),
            *self.witness_high.register(),
        ]
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        let mut set = HashSet::new();
        set.insert(*self.a.register());
        set
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let p_a = writer.read(&self.a, row_index);
        let mut c = BigUint::zero();
        for (i, limb) in self.c.iter().enumerate() {
            c += BigUint::from(*limb) << (16 * i);
        }

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let a = digits_to_biguint(&a_digits);

        // Compute field addition in the integers.
        let modulus = P::modulus();
        let result = (&a * &c) % &modulus;
        let carry = (&a * &c - &result) / &modulus;
        debug_assert!(result < modulus);
        debug_assert!(carry < modulus);
        debug_assert_eq!(&carry * &modulus, a * &c - &result);

        // Make little endian polynomial limbs.
        let p_c = to_u16_le_limbs_polynomial::<F, P>(&c);
        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(&modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let p_vanishing = &p_a * &p_c - &p_result - &p_carry * &p_modulus;
        debug_assert_eq!(p_vanishing.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness = util::compute_root_quotient_and_shift(&p_vanishing, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs(&p_witness);

        let mut values = p_result.coefficients;
        values.extend_from_slice(p_carry.coefficients());
        values.extend_from_slice(&p_witness_low);
        values.extend_from_slice(&p_witness_high);

        // Row must match layout of instruction.
        writer.write_unsafe_batch_raw(
            &vec![
                *self.result.register(),
                *self.carry.register(),
                *self.witness_low.register(),
                *self.witness_high.register(),
            ],
            &values,
            row_index,
        );
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
    struct FpMulConstTest;

    impl const AirParameters for FpMulConstTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 140;
        const NUM_FREE_COLUMNS: usize = 218;

        type Instruction = FpMulConstInstruction<Fp25519>;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_fpmul_const() {
        type F = GoldilocksField;
        type L = FpMulConstTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;

        let p = Fp25519::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc::<FieldRegister<P>>();

        let mut c: [u16; MAX_NB_LIMBS] = [0; MAX_NB_LIMBS];
        c[0] = 100;
        c[1] = 2;
        c[2] = 30000;

        let mul_const_insr = builder.fp_mul_const(&a, c);

        let (air, _) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&[]);

        let (tx, rx) = channel();

        let mut rng = thread_rng();
        for i in 0..L::num_rows() {
            let writer = generator.new_writer();
            let handle = tx.clone();
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            rayon::spawn(move || {
                let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
                writer.write(&a, p_a.coefficients(), i);
                writer.write_instruction(&mul_const_insr, i);

                handle.send(1).unwrap();
            });
        }
        drop(tx);
        for msg in rx.iter() {
            assert!(msg == 1);
        }
        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
