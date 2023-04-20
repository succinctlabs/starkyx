//! Implements field addition for any field, using a trick from Polygon Zero.
//! Reference: https://github.com/mir-protocol/plonky2/blob/main/evm/src/arithmetic/addcy.rs
//!
//! We want to compute a + b = result mod p. In the integers, this is equivalent to witnessing some
//! carry such that:
//!
//!     a + b - result - carry * p = 0.
//!
//! Let us encode the integers as polynomials in the Goldilocks field, where each coefficient is
//! at most 16 bits. In other words, the integers are encoded as an array of little-endian base 16
//! limbs. We can then write the above equation as:
//!
//!    a(x) + b(x) - result(x) - carry(x) * p(x)
//!
//! where the polynomial should evaluate to 0 if x = 2^16. To prove that the polynomial has a root
//! at 2^16, we can have the prover witness a polynomial `w(x)` such that the above polynomial
//! is divisble by (x - 2^16):
//!
//!
//!    a(x) + b(x) - result(x) - carry(x) * p(x) - (x - 2^16) * w(x) = 0
//!
//! Thus, if we can prove that above polynomial is 0, we can conclude that the addition has been
//! computed correctly. Note that this relies on the fact that any quadratic sum of a sufficiently
//! small number of terms (i.e., less than 2^32 terms) will not overflow in the Goldilocks field.
//! Furthermore, one must be careful to ensure that all polynomials except w(x) are range checked
//! in [0, 2^16).

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::builder::StarkBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::instruction::Instruction;
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::arithmetic::register::{
    ArrayRegister, FieldRegister, MemorySlice, RegisterSerializable, U16Register,
};
use crate::arithmetic::trace::TraceWriter;
use crate::arithmetic::utils::{extract_witness_and_shift, split_digits, to_field_iter};
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

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub fn fpadd<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
    ) -> Result<(FieldRegister<P>, FpAddInstruction<P>)>
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
        self.insert_instruction(instr.into())?;
        Ok((result, instr))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    pub fn write_fpadd<P: FieldParameters>(
        &self,
        row_index: usize,
        a: &BigUint,
        b: &BigUint,
        instruction: FpAddInstruction<P>,
    ) -> Result<BigUint> {
        let p = P::modulus();
        let result = (a + b) % &p;
        let carry = (a + b - &result) / &p;
        debug_assert!(result < p);
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, a + b - &result);

        // Make polynomial limbs.
        let p_a = Polynomial::<i64>::from_biguint_num(a, 16, P::NB_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(b, 16, P::NB_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, P::NB_LIMBS);
        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, P::NB_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, P::NB_LIMBS);

        // Compute the vanishing polynomial.
        let vanishing_poly = &p_a + &p_b - &p_result - &p_carry * &p_p;
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, P::WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        // Row must match layout of instruction.
        self.write_to_layout(
            row_index,
            instruction,
            vec![
                to_field_iter::<F>(&p_result).collect(),
                to_field_iter::<F>(&p_carry).collect(),
                witness_low.collect(),
                witness_high.collect(),
            ],
        )?;
        Ok(result)
    }
}

impl<F: RichField + Extendable<D>, const D: usize, P: FieldParameters> Instruction<F, D>
    for FpAddInstruction<P>
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
        // get all the data
        let a = self.a.register().packed_entries_slice(&vars);
        let b = self.b.register().packed_entries_slice(&vars);
        let result = self.result.register().packed_entries_slice(&vars);

        let carry = self.carry.register().packed_entries_slice(&vars);
        let witness_low = self.witness_low.register().packed_entries_slice(&vars);
        let witness_high = self.witness_high.register().packed_entries_slice(&vars);

        // Construct the expected vanishing polynmial
        let a_plus_b = PolynomialOps::add(a, b);
        let a_plus_b_minus_result = PolynomialOps::sub(&a_plus_b, result);
        let p_limbs = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, P>());
        let mul_times_carry = PolynomialOps::scalar_poly_mul(carry, p_limbs.as_slice());
        let vanishing_poly = PolynomialOps::sub(&a_plus_b_minus_result, &mul_times_carry);

        // reconstruct witness

        let limb = FE::from_canonical_u32(LIMB);

        // Reconstruct and shift back the witness polynomial
        let w_shifted = witness_low
            .iter()
            .zip(witness_high.iter())
            .map(|(x, y)| *x + (*y * limb));

        let offset = FE::from_canonical_u32(P::WITNESS_OFFSET as u32);
        let w = w_shifted.map(|x| x - offset).collect::<Vec<PF>>();

        // Multiply by (x-2^16) and make the constraint
        let root_monomial: &[PF] = &[PF::from(-limb), PF::from(PF::Scalar::ONE)];
        let witness_times_root = PolynomialOps::mul(&w, root_monomial);

        for i in 0..vanishing_poly.len() {
            yield_constr.constraint(vanishing_poly[i] - witness_times_root[i]);
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // get all the data
        let a = self.a.register().evaluation_targets(&vars);
        let b = self.b.register().evaluation_targets(&vars);
        let result = self.result.register().evaluation_targets(&vars);

        let carry = self.carry.register().evaluation_targets(&vars);
        let witness_low = self.witness_low.register().evaluation_targets(&vars);
        let witness_high = self.witness_high.register().evaluation_targets(&vars);

        // Construct the expected vanishing polynmial
        let a_plus_b = PolynomialGadget::add_extension(builder, a, b);
        let a_plus_b_minus_result = PolynomialGadget::sub_extension(builder, &a_plus_b, result);
        let p_limbs = PolynomialGadget::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, P>().collect::<Vec<_>>()[..],
        );
        let mul_times_carry = PolynomialGadget::mul_extension(builder, carry, &p_limbs[..]);
        let vanishing_poly =
            PolynomialGadget::sub_extension(builder, &a_plus_b_minus_result, &mul_times_carry);

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
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::builder::StarkBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
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
    struct FpAddTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for FpAddTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = FpAddInstruction<Fp25519Param>;
    }

    #[test]
    fn test_fpadd() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<FpAddTest, F, D>;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut builder = StarkBuilder::<FpAddTest, F, D>::new();

        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();

        let (_, a_add_b_ins) = builder.fpadd(&a, &b).unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus();

        let mut timing = TimingTree::new("stark_proof", log::Level::Debug);

        let trace = timed!(timing, "generate trace", {
            let mut rng = thread_rng();
            for i in 0..num_rows {
                let a_int: BigUint = rng.gen_biguint(256) % &p;
                let b_int = rng.gen_biguint(256) % &p;
                //let handle = handle.clone();
                //rayon::spawn(move || {
                handle.write_field(i as usize, &a_int, a).unwrap();
                handle.write_field(i as usize, &b_int, b).unwrap();
                handle
                    .write_fpadd(i as usize, &a_int, &b_int, a_add_b_ins)
                    .unwrap();
                //});
            }
            drop(handle);

            generator.generate_trace(&chip, num_rows as usize).unwrap()
        });

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

        // Verify proof as a stark
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
