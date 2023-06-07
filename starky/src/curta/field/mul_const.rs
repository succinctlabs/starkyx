//! Implements non-native field multiplication with a constant as an "instruction".
//!
//! To understand the implementation, it may be useful to refer to `mod.rs`.

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::constraint::{
    eval_field_operation, ext_circuit_field_operation, packed_generic_field_operation,
};
use super::*;
use crate::curta::air::parser::AirParser;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::instruction::Instruction;
use crate::curta::parameters::MAX_NB_LIMBS;
use crate::curta::polynomial::parser::PolynomialParser;
use crate::curta::polynomial::{
    to_u16_le_limbs_polynomial, Polynomial, PolynomialGadget, PolynomialOps,
};
use crate::curta::register::{
    ArrayRegister, MemorySlice, Register, RegisterSerializable, U16Register,
};
use crate::curta::trace::writer::TraceWriter;
use crate::curta::utils::{compute_root_quotient_and_shift, split_u32_limbs_to_u16_limbs};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct FpMulConstInstruction<P: FieldParameters> {
    a: FieldRegister<P>,
    c: [u16; MAX_NB_LIMBS],
    pub result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Given a field element `a` and a scalar constant `c`, computes the product `a * c = result`.
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
        self.constrain_instruction(instr.into()).unwrap();
        instr
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    /// Writes a `FpMulConstInstruction` to the trace and returns the result.
    pub fn write_fpmul_const<P: FieldParameters>(
        &self,
        row_index: usize,
        a: &BigUint,
        instruction: FpMulConstInstruction<P>,
    ) -> Result<BigUint> {
        let modulus = P::modulus();
        let mut c = BigUint::zero();
        for (i, limb) in instruction.c.iter().enumerate() {
            c += BigUint::from(*limb) << (16 * i);
        }
        let result = (a * &c) % &modulus;
        let carry = (a * &c - &result) / &modulus;
        debug_assert!(result < modulus);
        debug_assert!(carry < modulus);
        debug_assert_eq!(&carry * &modulus, a * &c - &result);

        // Make little endian polynomial limbs.
        let p_a = to_u16_le_limbs_polynomial::<F, P>(a);
        let p_c = to_u16_le_limbs_polynomial::<F, P>(&c);
        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(&modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let vanishing_poly = &p_a * &p_c - &p_result - &p_carry * &p_modulus;
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness_shifted = compute_root_quotient_and_shift(&vanishing_poly, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs::<F>(&p_witness_shifted);

        // Row must match layout of instruction.
        self.write_to_layout(
            row_index,
            instruction,
            vec![
                p_result.coefficients,
                p_carry.coefficients,
                p_witness_low,
                p_witness_high,
            ],
        )?;
        Ok(result)
    }
}

impl<F: RichField + Extendable<D>, const D: usize, P: FieldParameters> Instruction<F, D>
    for FpMulConstInstruction<P>
{
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.result.register(),
            *self.carry.register(),
            *self.witness_low.register(),
            *self.witness_high.register(),
        ]
    }

    fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var> {
        let mut poly_parser = PolynomialParser::new(parser);

        let p_a = self.a.eval(poly_parser.parser);
        let p_c = self
            .c
            .iter()
            .map(|c| F::from_canonical_u16(*c))
            .take(P::NB_LIMBS)
            .collect::<Polynomial<F>>();
        let p_result = self.result.eval(poly_parser.parser);
        let p_carry = self.carry.eval(poly_parser.parser);

        // Compute the vanishing polynomial a(x) * c(x) - result(x) - carry(x) * p(x).
        let p_a_mul_c = poly_parser.scalar_poly_mul(&p_a, &p_c);
        let p_a_mul_c_minus_result = poly_parser.sub(&p_a_mul_c, &p_result);
        let p_modulus = Polynomial::<F>::from_iter(modulus_field_iter::<F, P>());

        let p_carry_mul_modulus = poly_parser.scalar_poly_mul(&p_carry, &p_modulus);
        let p_vanishing = poly_parser.sub(&p_a_mul_c_minus_result, &p_carry_mul_modulus);

        let p_witness_low =
            Polynomial::from_coefficients(self.witness_low.eval(poly_parser.parser));
        let p_witness_high =
            Polynomial::from_coefficients(self.witness_high.eval(poly_parser.parser));

        eval_field_operation::<AP, P>(
            &mut poly_parser,
            &p_vanishing,
            &p_witness_low,
            &p_witness_high,
        )
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use rand::thread_rng;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{ChipStark, StarkParameters};
    use crate::curta::extension::cubic::goldilocks_cubic::GoldilocksCubicParameters;
    use crate::curta::parameters::ed25519::Ed25519BaseField;
    use crate::curta::stark::prover::prove;
    use crate::curta::stark::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::curta::stark::verifier::verify_stark_proof;
    use crate::curta::trace::arithmetic::{trace, ArithmeticGenerator};

    #[derive(Clone, Debug, Copy)]
    struct FpMulConstTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for FpMulConstTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 108;
        const NUM_FREE_COLUMNS: usize = 170;
        type Instruction = FpMulConstInstruction<Ed25519BaseField>;
    }

    #[test]
    fn test_fpmul_const() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = FieldRegister<Ed25519BaseField>;
        type E = GoldilocksCubicParameters;
        type S = ChipStark<FpMulConstTest, F, D>;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("stark_proof", log::Level::Debug);

        // Build the circuit.
        let mut c: [u16; MAX_NB_LIMBS] = [0; MAX_NB_LIMBS];
        c[0] = 100;
        c[1] = 2;
        c[2] = 30000;
        let mut c_bigint = BigUint::zero();
        for i in 0..MAX_NB_LIMBS {
            c_bigint += BigUint::from(c[i]) << (i * 16);
        }

        let mut builder = StarkBuilder::<FpMulConstTest, F, D>::new();
        let a = builder.alloc::<Fp>();
        let ac_ins = builder.fp_mul_const(&a, c);
        builder.write_data(&a).unwrap();
        let chip = builder.build();

        // Generate the trace.
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, E, D>(num_rows);
        let p = Ed25519BaseField::modulus();
        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            handle.write_field(i, &a_int, a).unwrap();
            let res = handle.write_fpmul_const(i, &a_int, ac_ins).unwrap();
            assert_eq!(res, (c_bigint.clone() * a_int) % &p);
        }

        drop(handle);

        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);
        let proof = prove::<F, C, S, ArithmeticGenerator<F, E, D>, D, 2>(
            stark.clone(),
            &config,
            generator,
            num_rows,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Generate the recursive proof.
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);
        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof = add_virtual_stark_proof_with_pis(
            &mut recursive_builder,
            stark.clone(),
            &config,
            degree_bits,
        );
        recursive_builder.print_gate_counts(0);
        let mut rec_pw = PartialWitness::new();
        set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);
        verify_stark_proof_circuit::<F, C, S, D, 2>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );
        let recursive_data = recursive_builder.build::<C>();
        let recursive_proof = timed!(
            timing,
            "generate recursive proof",
            plonky2::plonk::prover::prove(
                &recursive_data.prover_only,
                &recursive_data.common,
                rec_pw,
                &mut TimingTree::default(),
            )
            .unwrap()
        );
        timed!(
            timing,
            "verify recursive proof",
            recursive_data.verify(recursive_proof).unwrap()
        );
        timing.print();
    }
}
