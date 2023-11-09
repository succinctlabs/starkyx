use serde::{Deserialize, Serialize};

use super::add::FpAddInstruction;
use super::parameters::FieldParameters;
use super::register::FieldRegister;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::u16::U16Register;
use crate::chip::register::RegisterSerializable;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::utils::digits_to_biguint;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::to_u16_le_limbs_polynomial;

/// Fp subtraction.
///
/// prove a - b = c by asserting b + c = a.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FpSubInstruction<P: FieldParameters> {
    inner: FpAddInstruction<P>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// given two field elements `a` and `b`, computes the difference `a - b = c`.
    pub fn fp_sub<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
    ) -> FieldRegister<P>
    where
        L::Instruction: From<FpSubInstruction<P>>,
    {
        let is_trace = a.is_trace() || b.is_trace();
        let result = if is_trace {
            self.alloc::<FieldRegister<P>>()
        } else {
            self.alloc_public::<FieldRegister<P>>()
        };
        self.set_fp_sub(a, b, &result);
        result
    }

    pub fn set_fp_sub<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
        result: &FieldRegister<P>,
    ) where
        L::Instruction: From<FpSubInstruction<P>>,
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

        let inner_instr = FpAddInstruction {
            a: *result,
            b: *b,
            result: *a,
            carry,
            witness_low,
            witness_high,
        };

        let instr = FpSubInstruction { inner: inner_instr };
        if is_trace {
            self.register_instruction(instr);
        } else {
            self.register_global_instruction(instr);
        }
    }
}

impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpSubInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        self.inner.eval(parser);
    }
}

// Instruction trait
impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpSubInstruction<P> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let p_b = writer.read(&self.inner.b, row_index);
        let p_a = writer.read(&self.inner.result, row_index);

        let b_digits = p_b
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let b = digits_to_biguint(&b_digits);
        let a = digits_to_biguint(&a_digits);

        let modulus = P::modulus();
        let c = (&modulus + &a - &b) % &modulus;
        let p_c = to_u16_le_limbs_polynomial::<F, P>(&c);

        writer.write(&self.inner.a, &p_c, row_index);

        self.inner.write(writer, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let p_b = writer.read(&self.inner.b);
        let p_a = writer.read(&self.inner.result);

        let b_digits = p_b
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let a_digits = p_a
            .coefficients
            .iter()
            .map(|x| x.as_canonical_u64() as u16)
            .collect::<Vec<_>>();

        let b = digits_to_biguint(&b_digits);
        let a = digits_to_biguint(&a_digits);

        let modulus = P::modulus();
        let c = (&modulus + &a - &b) % &modulus;
        let p_c = to_u16_le_limbs_polynomial::<F, P>(&c);

        writer.write(&self.inner.a, &p_c);

        self.inner.write_to_air(writer);
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
    use crate::polynomial::Polynomial;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    struct FpSubTest;

    impl AirParameters for FpSubTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 140;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 219;

        type Instruction = FpSubInstruction<Fp25519>;
    }

    #[test]
    fn test_fpsub() {
        type F = GoldilocksField;
        type L = FpSubTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;

        let p = Fp25519::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a_pub = builder.alloc_public::<FieldRegister<P>>();
        let b_pub = builder.alloc_public::<FieldRegister<P>>();
        let _ = builder.fp_sub(&a_pub, &b_pub);

        let a = builder.alloc::<FieldRegister<P>>();
        let b = builder.alloc::<FieldRegister<P>>();
        let c = builder.fp_sub(&a, &b);
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
                let c_int = (&p + &a_int - &b_int) % &p;
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
