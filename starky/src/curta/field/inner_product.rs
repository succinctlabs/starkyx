//! Implements non-native inner product as an "instruction".
//!
//! To understand the implementation, it may be useful to refer to `mod.rs`.

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::constraint::{ext_circuit_field_operation, packed_generic_field_operation};
use super::*;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::instruction::Instruction;
use crate::curta::parameters::FieldParameters;
use crate::curta::polynomial::{
    to_u16_le_limbs_polynomial, Polynomial, PolynomialGadget, PolynomialOps,
};
use crate::curta::register::{ArrayRegister, MemorySlice, RegisterSerializable, U16Register};
use crate::curta::trace::TraceWriter;
use crate::curta::utils::{compute_root_quotient_and_shift, split_u32_limbs_to_u16_limbs};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone)]
pub struct FpInnerProductInstruction<P: FieldParameters> {
    a: Vec<FieldRegister<P>>,
    b: Vec<FieldRegister<P>>,
    pub result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub fn fp_inner_product<P: FieldParameters>(
        &mut self,
        a: &Vec<FieldRegister<P>>,
        b: &Vec<FieldRegister<P>>,
    ) -> FpInnerProductInstruction<P>
    where
        L::Instruction: From<FpInnerProductInstruction<P>>,
    {
        debug_assert!(a.len() == b.len());
        let result = self.alloc::<FieldRegister<P>>();
        let carry = self.alloc::<FieldRegister<P>>();
        let witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);

        let instr = FpInnerProductInstruction {
            a: a.clone(),
            b: b.clone(),
            result,
            carry,
            witness_low,
            witness_high,
        };
        self.constrain_instruction(instr.clone().into()).unwrap();
        instr
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    pub fn write_fp_inner_product<P: FieldParameters>(
        &self,
        row_index: usize,
        a: Vec<&BigUint>,
        b: Vec<&BigUint>,
        instruction: FpInnerProductInstruction<P>,
    ) -> Result<BigUint> {
        let modulus = &P::modulus();
        let inner_product = &a
            .clone()
            .into_iter()
            .zip(b.clone())
            .fold(BigUint::zero(), |acc, (c, d)| acc + c * d);
        let result = &(inner_product % modulus);
        let carry = &((inner_product - result) / modulus);
        assert!(result < modulus);
        assert!(carry < &(2u32 * modulus));
        assert_eq!(carry * modulus, inner_product - result);

        // Make little endian polynomial limbs.
        let p_a_vec = a
            .into_iter()
            .map(|a| to_u16_le_limbs_polynomial::<F, P>(&a))
            .collect::<Vec<Polynomial<F>>>();
        let p_b_vec = b
            .into_iter()
            .map(|b| to_u16_le_limbs_polynomial::<F, P>(&b))
            .collect::<Vec<Polynomial<F>>>();
        let p_modulus = to_u16_le_limbs_polynomial::<F, P>(&modulus);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let p_inner_product = p_a_vec.into_iter().zip(p_b_vec.into_iter()).fold(
            Polynomial::<F>::new_from_vec(vec![F::ZERO]),
            |acc, (c, d)| acc + &c * &d,
        );
        let p_vanishing = p_inner_product - &p_result - &p_carry * &p_modulus;
        assert_eq!(p_vanishing.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness
        let p_witness_shifted = compute_root_quotient_and_shift(&p_vanishing, P::WITNESS_OFFSET);
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
        Ok(result.clone())
    }
}

impl<F: RichField + Extendable<D>, const D: usize, P: FieldParameters> Instruction<F, D>
    for FpInnerProductInstruction<P>
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
        // Get packed entries.
        let p_a_vec = self
            .a
            .clone()
            .into_iter()
            .map(|x| x.register().packed_generic_vars(vars))
            .collect::<Vec<_>>();
        let p_b_vec = self
            .b
            .clone()
            .into_iter()
            .map(|x| x.register().packed_generic_vars(vars))
            .collect::<Vec<_>>();
        let p_result = self.result.register().packed_generic_vars(vars);
        let p_carry = self.carry.register().packed_generic_vars(vars);
        let p_witness_low = self.witness_low.register().packed_generic_vars(vars);
        let p_witness_high = self.witness_high.register().packed_generic_vars(vars);

        // Construct the expected vanishing polynmial.
        let p_zero = vec![PF::ZEROS];
        let p_inner_product = p_a_vec
            .into_iter()
            .zip(p_b_vec)
            .map(|(a, b)| PolynomialOps::mul(a, b))
            .fold(p_zero, |acc, x| PolynomialOps::add(&acc, &x[..]));
        let p_inner_product_minus_result = PolynomialOps::sub(&p_inner_product, p_result);
        let p_limbs = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, P>());
        let mul_times_carry = PolynomialOps::scalar_poly_mul(p_carry, p_limbs.as_slice());
        let p_vanishing = PolynomialOps::sub(&p_inner_product_minus_result, &mul_times_carry);

        // Check [(\sum_i a_i(x) + b_i(x)) - result(x) - carry(x) * p(x)] - [witness(x) * (x-2^16)] = 0.
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
        // Get the packed entries.
        let p_a_vec = self
            .a
            .clone()
            .into_iter()
            .map(|x| x.register().ext_circuit_vars(vars))
            .collect::<Vec<_>>();
        let p_b_vec = self
            .b
            .clone()
            .into_iter()
            .map(|x| x.register().ext_circuit_vars(vars))
            .collect::<Vec<_>>();
        let p_result = self.result.register().ext_circuit_vars(vars);
        let p_carry = self.carry.register().ext_circuit_vars(vars);
        let p_witness_low = self.witness_low.register().ext_circuit_vars(vars);
        let p_witness_high = self.witness_high.register().ext_circuit_vars(vars);

        // Construct the expected vanishing polynmial
        let p_zero = vec![builder.zero_extension()];
        let p_inner_product = p_a_vec
            .into_iter()
            .zip(p_b_vec)
            .map(|(a, b)| PolynomialGadget::mul_extension(builder, a, b))
            .collect::<Vec<_>>()
            .into_iter()
            .fold(p_zero, |acc, x| {
                PolynomialGadget::add_extension(builder, &acc, &x[..])
            });
        let p_inner_product_minus_result =
            PolynomialGadget::sub_extension(builder, &p_inner_product, p_result);
        let p_modulus = PolynomialGadget::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, P>().collect::<Vec<_>>(),
        );
        let p_carry_mul_modulus = PolynomialGadget::mul_extension(builder, p_carry, &p_modulus[..]);
        let p_vanishing = PolynomialGadget::sub_extension(
            builder,
            &p_inner_product_minus_result,
            &p_carry_mul_modulus,
        );

        // Check [(\sum_i a_i(x) + b_i(x)) - result(x) - carry(x) * p(x)] - [witness(x) * (x-2^16)] = 0.
        ext_circuit_field_operation::<F, D, P>(builder, p_vanishing, p_witness_low, p_witness_high)
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    //use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{ChipStark, StarkParameters};
    use crate::curta::parameters::ed25519::Ed25519BaseField;
    use crate::curta::trace::trace;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    struct FpInnerProductTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for FpInnerProductTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 156;
        const NUM_FREE_COLUMNS: usize = 0;
        type Instruction = FpInnerProductInstruction<Ed25519BaseField>;
    }

    #[test]
    fn test_fpquad() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = FieldRegister<Ed25519BaseField>;
        type S = ChipStark<FpInnerProductTest, F, D>;

        // Construct the circuit.
        let mut builder = StarkBuilder::<FpInnerProductTest, F, D>::new();
        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();
        let c = builder.alloc::<Fp>();
        let d = builder.alloc::<Fp>();
        let quad = builder.fp_inner_product(&vec![a, b], &vec![c, d]);
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();
        builder.write_data(&c).unwrap();
        builder.write_data(&d).unwrap();
        let (chip, spec) = builder.build();

        // Construct the trace.
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);
        let p = Ed25519BaseField::modulus();
        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let c_int = rng.gen_biguint(256) % &p;
            let d_int = rng.gen_biguint(256) % &p;
            let handle = handle.clone();
            handle.write_field(i as usize, &a_int, a).unwrap();
            handle.write_field(i as usize, &b_int, b).unwrap();
            handle.write_field(i as usize, &c_int, c).unwrap();
            handle.write_field(i as usize, &d_int, d).unwrap();
            let result = handle
                .write_fp_inner_product(
                    i as usize,
                    vec![&a_int, &b_int],
                    vec![&c_int, &d_int],
                    quad.clone(),
                )
                .unwrap();
            assert_eq!(result, (a_int * c_int + b_int * d_int) % &p);
        }
        drop(handle);
        let trace = generator.generate_trace(&chip, num_rows as usize).unwrap();

        // Construct the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);
        let proof = prove::<F, C, S, D>(
            stark.clone(),
            &config,
            trace,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit.
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
        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut timing,
        )
        .unwrap();
        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
