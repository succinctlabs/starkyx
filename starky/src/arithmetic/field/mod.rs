mod add;
mod mul;
mod mul_const;
mod quad;

use anyhow::Result;
use num::{BigUint, One, Zero};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness;

pub use self::add::FpAddInstruction;
pub use self::mul::FpMulInstruction;
pub use self::mul_const::FpMulConstInstruction;
pub use self::quad::FpQuadInstruction;
use super::instruction::Instruction;
use super::polynomial::Polynomial;
use super::register::FieldRegister;
use super::trace::TraceWriter;
use crate::arithmetic::register::MemorySlice;

pub const MAX_NB_LIMBS: usize = 32;
pub const LIMB: u32 = 2u32.pow(16);

pub trait FieldParameters: Send + Sync + Copy + 'static {
    const NB_BITS_PER_LIMB: usize;
    const NB_LIMBS: usize;
    const NB_WITNESS_LIMBS: usize;
    const MODULUS: [u16; MAX_NB_LIMBS];
    const WITNESS_OFFSET: usize;

    fn modulus() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::MODULUS.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }
}

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

/// The parameters for the Fp25519 field of modulues 2^255-19.
#[derive(Debug, Clone, Copy)]
pub struct Fp25519Param;

pub type Fp25519 = FieldRegister<Fp25519Param>;

impl FieldParameters for Fp25519Param {
    const NB_BITS_PER_LIMB: usize = 16;
    const NB_LIMBS: usize = 16;
    const NB_WITNESS_LIMBS: usize = 2 * Self::NB_LIMBS - 2;
    const MODULUS: [u16; MAX_NB_LIMBS] = [
        65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535, 32767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const WITNESS_OFFSET: usize = 1usize << 20;

    fn modulus() -> BigUint {
        (BigUint::one() << 255) - BigUint::from(19u32)
    }
}

#[derive(Debug, Clone)]
pub enum FpInstruction<P: FieldParameters> {
    Add(FpAddInstruction<P>),
    Mul(FpMulInstruction<P>),
    Quad(FpQuadInstruction<P>),
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

impl<P: FieldParameters> From<FpQuadInstruction<P>> for FpInstruction<P> {
    fn from(quad: FpQuadInstruction<P>) -> Self {
        Self::Quad(quad)
    }
}

impl<F: RichField + Extendable<D>, const D: usize, P: FieldParameters> Instruction<F, D>
    for FpInstruction<P>
{
    fn witness_layout(&self) -> Vec<MemorySlice> {
        match self {
            FpInstruction::Add(add) => {
                <FpAddInstruction<P> as Instruction<F, D>>::witness_layout(add)
            }
            FpInstruction::Mul(mul) => {
                <FpMulInstruction<P> as Instruction<F, D>>::witness_layout(mul)
            }
            FpInstruction::Quad(quad) => {
                <FpQuadInstruction<P> as Instruction<F, D>>::witness_layout(quad)
            }
            FpInstruction::MulConst(mul_const) => {
                <FpMulConstInstruction<P> as Instruction<F, D>>::witness_layout(mul_const)
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
                <FpQuadInstruction<P> as Instruction<F, D>>::packed_generic_constraints(
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
                <FpQuadInstruction<P> as Instruction<F, D>>::ext_circuit_constraints(
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
    struct FpInstructionTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for FpInstructionTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 156;
        const NUM_FREE_COLUMNS: usize = 0;
        type Instruction = FpInstruction<Fp25519Param>;
    }

    #[test]
    fn test_instructions_fpquad() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<FpInstructionTest, F, D>;

        // build the stark
        let mut builder = StarkBuilder::<FpInstructionTest, F, D>::new();

        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();
        let c = builder.alloc::<Fp>();
        let d = builder.alloc::<Fp>();

        let (result, quad) = builder.fpquad(&a, &b, &c, &d).unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();
        builder.write_data(&c).unwrap();
        builder.write_data(&d).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let c_int = rng.gen_biguint(256) % &p;
            let d_int = rng.gen_biguint(256) % &p;
            //let handle = handle.clone();
            //rayon::spawn(move || {
            handle.write_field(i as usize, &a_int, a).unwrap();
            handle.write_field(i as usize, &b_int, b).unwrap();
            handle.write_field(i as usize, &c_int, c).unwrap();
            handle.write_field(i as usize, &d_int, d).unwrap();
            handle
                .write_fpquad(i as usize, &a_int, &b_int, &c_int, &d_int, quad)
                .unwrap();
            //});
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows as usize).unwrap();

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
