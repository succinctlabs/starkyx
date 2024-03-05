use num::BigUint;
use serde::{Deserialize, Serialize};

use super::mul::FpMulInstruction;
use super::parameters::FieldParameters;
use super::register::FieldRegister;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
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

/// Fp Division. Computes `a / b = result`.
///
/// This is done by computing `b_inv = b^(-1)` followed by `a * b_inv = result`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FpDivInstruction<P: FieldParameters> {
    /// a `FpMulInstruction` to compute `b_inv = b^(-1)`.
    denominator: FpMulInstruction<P>,
    /// a `FpMulInstruction` to compute `a * b_inv = result`.
    multiplication: FpMulInstruction<P>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// given two field elements `a` and `b`, computes the quotient `a / b = c`.
    pub fn fp_div<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
    ) -> FieldRegister<P>
    where
        L::Instruction: From<FpDivInstruction<P>>,
    {
        let is_trace = a.is_trace() || b.is_trace();
        let result = if is_trace {
            self.alloc::<FieldRegister<P>>()
        } else {
            self.alloc_public::<FieldRegister<P>>()
        };
        self.set_fp_div(a, b, &result);
        result
    }

    pub fn set_fp_div<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
        result: &FieldRegister<P>,
    ) where
        L::Instruction: From<FpDivInstruction<P>>,
    {
        let is_trace = a.is_trace() || b.is_trace() || result.is_trace();

        let denom_carry: FieldRegister<P>;
        let denom_witness_low: ArrayRegister<U16Register>;
        let denom_witness_high: ArrayRegister<U16Register>;
        let one = self.fp_one();
        let b_inv: FieldRegister<P>;

        if is_trace {
            denom_carry = self.alloc::<FieldRegister<P>>();
            denom_witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
            denom_witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
            b_inv = self.alloc::<FieldRegister<P>>();
        } else {
            denom_carry = self.alloc_public::<FieldRegister<P>>();
            denom_witness_low = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
            denom_witness_high = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
            b_inv = self.alloc_public::<FieldRegister<P>>();
        }

        // check that b * b_inv = one.
        let denominator = FpMulInstruction {
            a: *b,
            b: b_inv,
            result: one,
            carry: denom_carry,
            witness_low: denom_witness_low,
            witness_high: denom_witness_high,
        };

        let mult_carry: FieldRegister<P>;
        let mult_witness_low: ArrayRegister<U16Register>;
        let mult_witness_high: ArrayRegister<U16Register>;
        if is_trace {
            mult_carry = self.alloc::<FieldRegister<P>>();
            mult_witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
            mult_witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        } else {
            mult_carry = self.alloc_public::<FieldRegister<P>>();
            mult_witness_low = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
            mult_witness_high = self.alloc_array_public::<U16Register>(P::NB_WITNESS_LIMBS);
        }

        // set the instruction a * b_inv = result.
        let multiplication = FpMulInstruction {
            a: *a,
            b: b_inv,
            result: *result,
            carry: mult_carry,
            witness_low: mult_witness_low,
            witness_high: mult_witness_high,
        };

        let instr = FpDivInstruction {
            denominator,
            multiplication,
        };

        if is_trace {
            self.register_instruction(instr);
        } else {
            self.register_global_instruction(instr);
        }
    }
}

impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpDivInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        self.denominator.eval(parser);
        self.multiplication.eval(parser);
    }
}

impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpDivInstruction<P> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let p_b = writer.read(&self.denominator.a, row_index);

        let b_digits = p_b
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let b = digits_to_biguint(&b_digits);

        let modulus = P::modulus();
        let b_inv_int = b.modpow(&(&modulus - BigUint::from(2u64)), &modulus);
        let p_b_inv = to_u16_le_limbs_polynomial::<F, P>(&b_inv_int);

        let b_inv = &self.denominator.b;

        writer.write(b_inv, &p_b_inv, row_index);

        self.denominator.write(writer, row_index);
        self.multiplication.write(writer, row_index);
    }

    fn write_to_air(&self, writer: &mut impl crate::chip::trace::writer::AirWriter<Field = F>) {
        let p_b = writer.read(&self.denominator.a);

        let b_digits = p_b
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let b = digits_to_biguint(&b_digits);

        let modulus = P::modulus();
        let b_inv_int = b.modpow(&(&modulus - BigUint::from(2u64)), &modulus);
        let p_b_inv = to_u16_le_limbs_polynomial::<F, P>(&b_inv_int);

        let b_inv = &self.denominator.b;

        writer.write(b_inv, &p_b_inv);

        self.denominator.write_to_air(writer);
        self.multiplication.write_to_air(writer);
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::field::parameters::tests::Fp25519;
    use crate::polynomial::Polynomial;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    struct FpDivTest;

    impl AirParameters for FpDivTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 248;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 381;

        type Instruction = FpDivInstruction<Fp25519>;
    }

    #[test]
    fn test_fpdiv() {
        type F = GoldilocksField;
        type L = FpDivTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;

        let p = Fp25519::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a_pub = builder.alloc_public::<FieldRegister<P>>();
        let b_pub = builder.alloc_public::<FieldRegister<P>>();
        let _ = builder.fp_div(&a_pub, &b_pub);

        let a = builder.alloc::<FieldRegister<P>>();
        let b = builder.alloc::<FieldRegister<P>>();
        let c = builder.fp_div(&a, &b);
        let c_expected = builder.alloc::<FieldRegister<P>>();
        builder.assert_equal(&c, &c_expected);

        let (air, trace_data) = builder.build();
        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let trace_initial = (0..num_rows)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let writer = generator.new_writer();
                let a_int: BigUint = rng.gen_biguint(256) % &p;
                let b_int = rng.gen_biguint(256) % &p;
                let b_inv_int = b_int.modpow(&(&p - BigUint::from(2u32)), &p);
                let c_int = (&a_int * &b_inv_int) % &p;
                (writer, a_int, b_int, c_int)
            })
            .collect::<Vec<_>>();

        trace_initial
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (writer, a_int, b_int, c_int))| {
                let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
                let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);
                let p_c = Polynomial::<F>::from_biguint_field(&c_int, 16, 16);

                writer.write(&a, &p_a, i);
                writer.write(&b, &p_b, i);
                writer.write(&c_expected, &p_c, i);

                writer.write(&a_pub, &p_a, i);
                writer.write(&b_pub, &p_b, i);

                writer.write_row_instructions(&generator.air_data, i);
            });

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
