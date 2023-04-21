//! Implements non-native field multiplication with a constant as an "instruction".
//!
//! To understand the implementation, it may be useful to refer to `mod.rs`.

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::constrain::packed_generic_constrain_field_operation;
use super::*;
use crate::arithmetic::builder::StarkBuilder;
use crate::arithmetic::chip::StarkParameters;
use crate::arithmetic::instruction::Instruction;
use crate::arithmetic::polynomial::{
    to_u16_le_limbs_polynomial, Polynomial, PolynomialGadget, PolynomialOps,
};
use crate::arithmetic::register::{ArrayRegister, MemorySlice, RegisterSerializable, U16Register};
use crate::arithmetic::trace::TraceWriter;
use crate::arithmetic::utils::{compute_root_quotient_and_shift, split_u32_limbs_to_u16_limbs};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct FpMulConstInstruction<P: FieldParameters> {
    a: FieldRegister<P>,
    c: [u16; MAX_NB_LIMBS],
    result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Given a field element `a` and a scalar constant `c`, computes the product `a * c = result`.
    pub fn fpmul_const<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        c: [u16; MAX_NB_LIMBS],
    ) -> Result<(FieldRegister<P>, FpMulConstInstruction<P>)>
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
        self.insert_instruction(instr.into())?;
        Ok((result, instr))
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

        // Compute the vanishing polynomial
        let vanishing_poly = &p_a * &p_c - &p_result - &p_carry * &p_modulus;
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness
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
    fn witness_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.result.register(),
            *self.carry.register(),
            *self.witness_low.register(),
            *self.witness_high.register(),
        ]
    }

    fn packed_generic_constraints<
        FE,
        PF,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, PF, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<PF>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        PF: PackedField<Scalar = FE>,
    {
        // Get the packed entries.
        let p_a = self.a.register().packed_entries(&vars);
        let c = self
            .c
            .into_iter()
            .map(FE::from_canonical_u16)
            .map(PF::from)
            .take(P::NB_LIMBS)
            .collect::<Vec<_>>();
        let p_result = self.result.register().packed_entries(&vars);
        let p_carry = self.carry.register().packed_generic_vars(&vars);
        let p_witness_low = self.witness_low.register().packed_generic_vars(&vars);
        let p_witness_high = self.witness_high.register().packed_generic_vars(&vars);

        // Compute the vanishing polynomial a(x) * c - result(x) - carry(x) * p(x).
        let p_a_mul_c = PolynomialOps::mul(&p_a, &c);
        let p_a_mul_c_minus_result = PolynomialOps::sub(&p_a_mul_c, &p_result);
        let p_modulus = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, P>());
        let p_carry_mul_modulus = PolynomialOps::scalar_poly_mul(p_carry, p_modulus.as_slice());
        let p_vanishing = PolynomialOps::sub(&p_a_mul_c_minus_result, &p_carry_mul_modulus);

        // Check [a(x) * c - result(x) - carry(x) * p(x)] - [witness(x) * (x-2^16)] = 0.
        packed_generic_constrain_field_operation::<F, D, FE, PF, D2, P>(
            yield_constr,
            p_vanishing,
            p_witness_low,
            p_witness_high,
        );
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // get all the data
        let a = self.a.register().ext_circuit_vars(&vars);
        let c_vec = self
            .c
            .into_iter()
            .map(F::Extension::from_canonical_u16)
            .take(P::NB_LIMBS)
            .collect::<Vec<_>>();
        let c = PolynomialGadget::constant_extension(builder, &c_vec);
        let result = self.result.register().ext_circuit_vars(&vars);

        let carry = self.carry.register().ext_circuit_vars(&vars);
        let witness_low = self.witness_low.register().ext_circuit_vars(&vars);
        let witness_high = self.witness_high.register().ext_circuit_vars(&vars);

        // Construct the expected vanishing polynmial
        let ac = PolynomialGadget::mul_extension(builder, a, &c);
        let ac_minus_result = PolynomialGadget::sub_extension(builder, &ac, result);
        let p_limbs = PolynomialGadget::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, P>().collect::<Vec<_>>(),
        );
        let mul_times_carry = PolynomialGadget::mul_extension(builder, carry, &p_limbs[..]);
        let vanishing_poly =
            PolynomialGadget::sub_extension(builder, &ac_minus_result, &mul_times_carry);

        // reconstruct witness

        // Reconstruct and shift back the witness polynomial
        let limb_const = F::Extension::from_canonical_u32(2u32.pow(16));
        let limb = builder.constant_extension(limb_const);
        let w_high_times_limb =
            PolynomialGadget::ext_scalar_mul_extension(builder, witness_high, &limb);
        let w_shifted = PolynomialGadget::add_extension(builder, witness_low, &w_high_times_limb);
        let offset =
            builder.constant_extension(F::Extension::from_canonical_u32(P::WITNESS_OFFSET as u32));
        let w = PolynomialGadget::sub_constant_extension(builder, &w_shifted, &offset);

        // Multiply by (x-2^16) and make the constraint
        let neg_limb = builder.constant_extension(-limb_const);
        let root_monomial = &[neg_limb, builder.constant_extension(F::Extension::ONE)];
        let witness_times_root =
            PolynomialGadget::mul_extension(builder, w.as_slice(), root_monomial);

        let constraint =
            PolynomialGadget::sub_extension(builder, &vanishing_poly, &witness_times_root);
        for constr in constraint {
            yield_constr.constraint(builder, constr);
        }
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
    use crate::arithmetic::builder::StarkBuilder;
    use crate::arithmetic::chip::{StarkParameters, TestStark};
    use crate::arithmetic::field::Fp25519Param;
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    struct FpMulTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for FpMulTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = FpMulInstruction<Fp25519Param>;
    }

    #[derive(Clone, Debug, Copy)]
    struct FpMulConstTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for FpMulConstTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 108;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = FpMulConstInstruction<Fp25519Param>;
    }

    #[test]
    fn test_fpmul_const() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<FpMulConstTest, F, D>;

        let mut c: [u16; MAX_NB_LIMBS] = [0; MAX_NB_LIMBS];
        c[0] = 100;
        c[1] = 2;
        c[2] = 30000;

        let mut c_bigint = BigUint::zero();
        for i in 0..MAX_NB_LIMBS {
            c_bigint += BigUint::from(c[i]) << (i * 16);
        }

        // build the stark
        let mut builder = StarkBuilder::<FpMulConstTest, F, D>::new();

        let a = builder.alloc::<Fp>();

        //let ab = FMul::new(a, b, result);
        //builder.insert_instruction(ab).unwrap();
        let (result, ac_ins) = builder.fpmul_const(&a, c).unwrap();
        builder.write_data(&a).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            //let handle = handle.clone();
            //rayon::spawn(move || {
            handle.write_field(i, &a_int, a).unwrap();
            let res = handle.write_fpmul_const(i, &a_int, ac_ins).unwrap();
            assert_eq!(res, (c_bigint.clone() * a_int) % &p);
            //});
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows).unwrap();

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

        // Verify proof as a stark
        let proof = prove::<F, C, S, D>(
            stark.clone(),
            &config,
            trace,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
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
