//! Implements field arithmetic for any field, using a trick from Polygon Zero.
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
//!
//! This technique generalizes for any quadratic sum with a "small" number of terms to avoid
//! overflow.

mod add;
mod constraint;
mod den;
mod inner_product;
mod mul;
mod mul_const;

use anyhow::Result;
use num::{BigUint, Zero};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

pub use self::add::FpAddInstruction;
pub use self::den::FpDenInstruction;
pub use self::inner_product::FpInnerProductInstruction;
pub use self::mul::FpMulInstruction;
pub use self::mul_const::FpMulConstInstruction;
use super::instruction::Instruction;
use super::parameters::FieldParameters;
use super::polynomial::Polynomial;
use super::register::FieldRegister;
use super::trace::TraceWriter;
use crate::curta::register::MemorySlice;

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    pub fn write_field<P: FieldParameters>(
        &self,
        row_index: usize,
        a_int: &BigUint,
        a: FieldRegister<P>,
    ) -> Result<()> {
        let p_a = Polynomial::<F>::from_biguint_field(a_int, 16, 16);
        self.write_data(row_index, a, p_a.into_vec())
    }
}

pub fn modulus_field_iter<F: Field, P: FieldParameters>() -> impl Iterator<Item = F> {
    P::MODULUS
        .into_iter()
        .map(|x| F::from_canonical_u16(x))
        .take(P::NB_LIMBS)
}

#[derive(Debug, Clone)]
pub enum FpInstruction<P: FieldParameters> {
    Add(FpAddInstruction<P>),
    Mul(FpMulInstruction<P>),
    Quad(FpInnerProductInstruction<P>),
    MulConst(FpMulConstInstruction<P>),
}

impl<P: FieldParameters> From<FpAddInstruction<P>> for FpInstruction<P> {
    fn from(add: FpAddInstruction<P>) -> Self {
        Self::Add(add)
    }
}

impl<P: FieldParameters> From<FpMulInstruction<P>> for FpInstruction<P> {
    fn from(mul: FpMulInstruction<P>) -> Self {
        Self::Mul(mul)
    }
}

impl<P: FieldParameters> From<FpInnerProductInstruction<P>> for FpInstruction<P> {
    fn from(quad: FpInnerProductInstruction<P>) -> Self {
        Self::Quad(quad)
    }
}

impl<F: RichField + Extendable<D>, const D: usize, P: FieldParameters> Instruction<F, D>
    for FpInstruction<P>
{
    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            FpInstruction::Add(add) => {
                <FpAddInstruction<P> as Instruction<F, D>>::trace_layout(add)
            }
            FpInstruction::Mul(mul) => {
                <FpMulInstruction<P> as Instruction<F, D>>::trace_layout(mul)
            }
            FpInstruction::Quad(quad) => {
                <FpInnerProductInstruction<P> as Instruction<F, D>>::trace_layout(quad)
            }
            FpInstruction::MulConst(mul_const) => {
                <FpMulConstInstruction<P> as Instruction<F, D>>::trace_layout(mul_const)
            }
        }
    }

    fn packed_generic_constraints<
        FE,
        PF,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: crate::vars::StarkEvaluationVars<FE, PF, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<PF>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        PF: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        match self {
            FpInstruction::Add(add) => {
                <FpAddInstruction<P> as Instruction<F, D>>::packed_generic_constraints(
                    add,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Mul(mul) => {
                <FpMulInstruction<P> as Instruction<F, D>>::packed_generic_constraints(
                    mul,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Quad(quad) => {
                <FpInnerProductInstruction<P> as Instruction<F, D>>::packed_generic_constraints(
                    quad,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::MulConst(mul_const) => {
                <FpMulConstInstruction<P> as Instruction<F, D>>::packed_generic_constraints(
                    mul_const,
                    vars,
                    yield_constr,
                )
            }
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: crate::vars::StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            FpInstruction::Add(add) => {
                <FpAddInstruction<P> as Instruction<F, D>>::ext_circuit_constraints(
                    add,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Mul(mul) => {
                <FpMulInstruction<P> as Instruction<F, D>>::ext_circuit_constraints(
                    mul,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Quad(quad) => {
                <FpInnerProductInstruction<P> as Instruction<F, D>>::ext_circuit_constraints(
                    quad,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::MulConst(mul_const) => {
                <FpMulConstInstruction<P> as Instruction<F, D>>::ext_circuit_constraints(
                    mul_const,
                    builder,
                    vars,
                    yield_constr,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    //use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{StarkParameters, TestStark};
    use crate::curta::parameters::ed25519::Ed25519BaseField;
    use crate::curta::trace::trace;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    struct FpInstructionTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for FpInstructionTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 156;
        const NUM_FREE_COLUMNS: usize = 0;
        type Instruction = FpInstruction<Ed25519BaseField>;
    }

    #[test]
    fn test_instructions_fpquad() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = FieldRegister<Ed25519BaseField>;
        type S = TestStark<FpInstructionTest, F, D>;

        // Build the STARK.
        let mut builder = StarkBuilder::<FpInstructionTest, F, D>::new();
        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();
        let c = builder.alloc::<Fp>();
        let d = builder.alloc::<Fp>();
        let (_, quad) = builder.fp_inner_product(&vec![a, b], &vec![c, d]);
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
            handle.write_field(i as usize, &a_int, a).unwrap();
            handle.write_field(i as usize, &b_int, b).unwrap();
            handle.write_field(i as usize, &c_int, c).unwrap();
            handle.write_field(i as usize, &d_int, d).unwrap();
            handle
                .write_fp_inner_product(
                    i as usize,
                    vec![&a_int, &b_int],
                    vec![&c_int, &d_int],
                    quad.clone(),
                )
                .unwrap();
        }
        drop(handle);
        let trace = generator.generate_trace(&chip, num_rows as usize).unwrap();

        // Verify proof as a stark.
        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);
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
