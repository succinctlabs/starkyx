//! Implements non-native field addition as an "instruction".
//!
//! To understand the implementation, it may be useful to refer to `mod.rs`.

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::constraint::packed_generic_field_operation;
use super::*;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::field::constraint::ext_circuit_field_operation;
use crate::curta::instruction::Instruction;
use crate::curta::parameters::FieldParameters;
use crate::curta::polynomial::{
    to_u16_le_limbs_polynomial, Polynomial, PolynomialGadget, PolynomialOps,
};
use crate::curta::register::{
    ArrayRegister, FieldRegister, MemorySlice, RegisterSerializable, U16Register,
};
use crate::curta::trace::TraceWriter;
use crate::curta::utils::{compute_root_quotient_and_shift, split_u32_limbs_to_u16_limbs};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct FpAddInstruction<P: FieldParameters> {
    a: FieldRegister<P>,
    b: FieldRegister<P>,
    result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<P: FieldParameters> FpAddInstruction<P> {
    pub fn set_inputs(&mut self, a: &FieldRegister<P>, b: &FieldRegister<P>) {
        self.a = *a;
        self.b = *b;
    }
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Given two field elements `a` and `b`, computes the sum `a + b = c`.
    pub fn fp_add<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
    ) -> Result<FpAddInstruction<P>>
    where
        L::Instruction: From<FpAddInstruction<P>>,
    {
        let result = self.alloc::<FieldRegister<P>>();
        let carry = self.alloc::<FieldRegister<P>>();
        let witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let instr = FpAddInstruction {
            a: *a,
            b: *b,
            result,
            carry,
            witness_low,
            witness_high,
        };
        self.constrain_instruction(instr.into())?;
        Ok(instr)
    }

    pub fn alloc_fp_add_instruction<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
    ) -> Result<FpAddInstruction<P>>
    where
        L::Instruction: From<FpAddInstruction<P>>,
    {
        let result = self.alloc::<FieldRegister<P>>();
        let carry = self.alloc::<FieldRegister<P>>();
        let witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let instr = FpAddInstruction {
            a: *a,
            b: *b,
            result,
            carry,
            witness_low,
            witness_high,
        };
        Ok(instr)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    /// Writes a `FpAddInstruction` to the trace and returns the result.
    pub fn write_fp_add<P: FieldParameters>(
        &self,
        row_index: usize,
        a: &BigUint,
        b: &BigUint,
        instruction: FpAddInstruction<P>,
    ) -> Result<BigUint> {
        // Compute field addition in the integers.
        let modulus = P::modulus();
        let result = (a + b) % &modulus;
        let carry = (a + b - &result) / &modulus;
        debug_assert!(result < modulus);
        debug_assert!(carry < modulus);
        debug_assert_eq!(&carry * &modulus, a + b - &result);

        // Make little endian polynomial limbs.
        let p_a = to_u16_le_limbs_polynomial::<F, P>(a);
        let p_b = to_u16_le_limbs_polynomial::<F, P>(b);
        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(&modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let p_vanishing = &p_a + &p_b - &p_result - &p_carry * &p_modulus;
        debug_assert_eq!(p_vanishing.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness = compute_root_quotient_and_shift(&p_vanishing, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs(&p_witness);

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
    for FpAddInstruction<P>
{
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.result.register(),
            *self.carry.register(),
            *self.witness_low.register(),
            *self.witness_high.register(),
        ]
    }

    fn packed_generic<FE, PF, const D2: usize, const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        vars: StarkEvaluationVars<FE, PF, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<PF>
    where
        FE: FieldExtension<D2, BaseField = F>,
        PF: PackedField<Scalar = FE>,
    {
        // Get the packed entries.
        let p_a = self.a.register().packed_generic_vars(vars);
        let p_b = self.b.register().packed_generic_vars(vars);
        let p_result = self.result.register().packed_generic_vars(vars);
        let p_carry = self.carry.register().packed_generic_vars(vars);
        let p_witness_low = self.witness_low.register().packed_generic_vars(vars);
        let p_witness_high = self.witness_high.register().packed_generic_vars(vars);

        // Compute the vanishing polynomial a(x) + b(x) - result(x) - carry(x) * p(x).
        let p_a_plus_b = PolynomialOps::add(p_a, p_b);
        let p_a_plus_b_minus_result = PolynomialOps::sub(&p_a_plus_b, p_result);
        let p_modulus = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, P>());
        let p_carry_mul_modulus = PolynomialOps::scalar_poly_mul(p_carry, p_modulus.as_slice());
        let p_vanishing = PolynomialOps::sub(&p_a_plus_b_minus_result, &p_carry_mul_modulus);

        // Check [a(x) + b(x) - result(x) - carry(x) * p(x)] - [witness(x) * (x-2^16)] = 0.
        packed_generic_field_operation::<F, D, FE, PF, D2, P>(
            p_vanishing,
            p_witness_low,
            p_witness_high,
        )
    }

    fn ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<ExtensionTarget<D>> {
        type PG = PolynomialGadget;

        // Get the packed entries.
        let p_a = self.a.register().ext_circuit_vars(vars);
        let p_b = self.b.register().ext_circuit_vars(vars);
        let p_result = self.result.register().ext_circuit_vars(vars);
        let p_carry = self.carry.register().ext_circuit_vars(vars);
        let p_witness_low = self.witness_low.register().ext_circuit_vars(vars);
        let p_witness_high = self.witness_high.register().ext_circuit_vars(vars);

        // Compute the vanishing polynomial a(x) + b(x) - result(x) - carry(x) * p(x).
        let p_a_plus_b = PG::add_extension(builder, p_a, p_b);
        let p_a_plus_b_minus_result = PG::sub_extension(builder, &p_a_plus_b, p_result);
        let p_limbs = PG::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, P>().collect::<Vec<_>>()[..],
        );
        let p_mul_times_carry = PG::mul_extension(builder, p_carry, &p_limbs[..]);
        let p_vanishing = PG::sub_extension(builder, &p_a_plus_b_minus_result, &p_mul_times_carry);

        ext_circuit_field_operation::<F, D, P>(builder, p_vanishing, p_witness_low, p_witness_high)
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
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{StarkParameters, TestStark};
    use crate::curta::constraint::arithmetic::ArithmeticExpression;
    use crate::curta::constraint::expression::ConstraintExpression;
    use crate::curta::extension::cubic::goldilocks_cubic::GoldilocksCubicParameters;
    use crate::curta::instruction::InstructionSet;
    use crate::curta::parameters::ed25519::{Ed25519, Ed25519BaseField};
    use crate::curta::parameters::EllipticCurveParameters;
    use crate::curta::register::{BitRegister, Register};
    use crate::curta::trace::{trace, trace_new};
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    struct FpAddTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for FpAddTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 140;
        const NUM_FREE_COLUMNS: usize = 8;
        type Instruction = InstructionSet<Ed25519BaseField>;
    }

    #[test]
    fn test_fpadd() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<FpAddTest, F, D>;
        type P = Ed25519BaseField;
        type E = GoldilocksCubicParameters;
        type L = FpAddTest;

        // Build the circuit.
        let mut builder = StarkBuilder::<FpAddTest, F, D>::new();
        let a = builder.alloc::<FieldRegister<P>>();
        let b = builder.alloc::<FieldRegister<P>>();
        let e = ArithmeticExpression::one();
        let a_add_b_ins = builder.alloc_fp_add_instruction(&a, &b).unwrap();
        let a_add_b_ins_s = InstructionSet::from(a_add_b_ins);
        let a_b_expr = a_add_b_ins_s.expr();
        builder.constraint(a_b_expr.clone() * e).unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();

        let range_data = builder.arithmetic_range_checks::<E>();

        let (chip, spec) = builder.build();

        // Generate the trace.
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace_new::<L, F, D>(&chip, num_rows, spec);
        let mut timing = TimingTree::new("stark_proof", log::Level::Debug);
        let trace = timed!(timing, "generate trace", {
            let p = <Ed25519 as EllipticCurveParameters>::BaseField::modulus();
            let mut rng = thread_rng();
            for i in 0..num_rows {
                let a_int: BigUint = rng.gen_biguint(256) % &p;
                let b_int = rng.gen_biguint(256) % &p;
                handle.write_field(i, &a_int, a).unwrap();
                handle.write_field(i, &b_int, b).unwrap();
                handle.write_fp_add(i, &a_int, &b_int, a_add_b_ins).unwrap();
            }
            drop(handle);
            generator
                .generate_trace_new::<L, E>(&chip, num_rows as usize)
                .unwrap()
        });

        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);
        let proof = timed!(
            timing,
            "generate proof",
            prove::<F, C, S, D>(
                stark.clone(),
                &config,
                trace,
                [],
                &mut TimingTree::default(),
            )
            .unwrap()
        );
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
        verify_stark_proof_circuit::<F, C, S, D>(
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

    #[test]
    fn test_expression_fpadd() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<FpAddTest, F, D>;
        type P = Ed25519BaseField;

        // Build the circuit.
        let mut builder = StarkBuilder::<FpAddTest, F, D>::new();
        let a = builder.alloc::<FieldRegister<P>>();
        let b = builder.alloc::<FieldRegister<P>>();
        let c = builder.alloc::<FieldRegister<P>>();

        let bit = builder.alloc::<BitRegister>();
        let a_add_b_ins = builder.alloc_fp_add_instruction(&a, &b).unwrap();
        let mut a_add_c_ins = a_add_b_ins.clone();
        a_add_c_ins.set_inputs(&a, &c);

        let a_add_b_expr = InstructionSet::from(a_add_b_ins).expr();
        let a_add_c_expr = InstructionSet::from(a_add_c_ins).expr();

        builder
            .constraint(a_add_b_expr * bit.expr() + a_add_c_expr * (bit.expr() - F::ONE))
            .unwrap();
        builder.constraint(ConstraintExpression::Empty).unwrap();

        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();
        builder.write_data(&c).unwrap();
        builder.write_data(&bit).unwrap();

        let (chip, spec) = builder.build();

        // Generate the trace.
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);
        let mut timing = TimingTree::new("stark_proof", log::Level::Debug);
        let trace = timed!(timing, "generate trace", {
            let p = <Ed25519 as EllipticCurveParameters>::BaseField::modulus();
            let mut rng = thread_rng();
            for i in 0..num_rows {
                let a_int: BigUint = rng.gen_biguint(256) % &p;
                let b_int = rng.gen_biguint(256) % &p;
                let c_int = rng.gen_biguint(256) % &p;
                let bit_val = rng.gen_bool(0.5);

                handle.write_field(i, &a_int, a).unwrap();
                handle.write_field(i, &b_int, b).unwrap();
                handle.write_field(i, &c_int, c).unwrap();
                handle.write_bit(i, bit_val, &bit).unwrap();

                if bit_val {
                    handle.write_fp_add(i, &a_int, &b_int, a_add_b_ins).unwrap();
                } else {
                    handle.write_fp_add(i, &a_int, &c_int, a_add_c_ins).unwrap();
                }
            }
            drop(handle);
            generator.generate_trace(&chip, num_rows as usize).unwrap()
        });

        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);
        let proof = timed!(
            timing,
            "generate proof",
            prove::<F, C, S, D>(
                stark.clone(),
                &config,
                trace,
                [],
                &mut TimingTree::default(),
            )
            .unwrap()
        );
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
        verify_stark_proof_circuit::<F, C, S, D>(
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
