use serde::{Deserialize, Serialize};

use super::add::FpAddInstruction;
use super::parameters::FieldParameters;
use super::register::FieldRegister;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::u16::U16Register;
use crate::chip::register::RegisterSerializable;
use crate::chip::trace::writer::TraceWriter;
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
        let result = self.alloc::<FieldRegister<P>>();
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
        let carry = self.alloc::<FieldRegister<P>>();
        let witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let inner_instr = FpAddInstruction {
            a: *result,
            b: *b,
            result: *a,
            carry,
            witness_low,
            witness_high,
        };

        let instr = FpSubInstruction { inner: inner_instr };
        self.register_instruction(instr);
    }
}

impl<AP: PolynomialParser, P: FieldParameters> AirConstraint<AP> for FpSubInstruction<P> {
    fn eval(&self, parser: &mut AP) {
        self.inner.eval(parser);
    }
}

// Instruction trait
impl<F: PrimeField64, P: FieldParameters> Instruction<F> for FpSubInstruction<P> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.inner.a.register(),
            *self.inner.carry.register(),
            *self.inner.witness_low.register(),
            *self.inner.witness_high.register(),
        ]
    }

    fn inputs(&self) -> Vec<MemorySlice> {
        vec![*self.inner.b.register(), *self.inner.result.register()]
    }

    fn constraint_degree(&self) -> usize {
        2
    }

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

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_fpsub() {
        type F = GoldilocksField;
        type L = FpSubTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;

        let p = Fp25519::modulus();

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc::<FieldRegister<P>>();
        let b = builder.alloc::<FieldRegister<P>>();
        let c = builder.fp_sub(&a, &b);
        let c_expected = builder.alloc::<FieldRegister<P>>();
        builder.assert_equal(&c, &c_expected);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);

        let trace_initial = (0..L::num_rows())
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
                writer.write_row_instructions(&generator.air_data, i);
            });

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
