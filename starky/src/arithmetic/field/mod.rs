pub mod add;
pub mod div;
pub mod mul;
pub mod quad;

use anyhow::Result;
use num::{BigUint, One, Zero};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use self::add::FpAdd;
use self::mul::FpMul;
use self::quad::FpQuad;
use super::instruction::Instruction;
use super::polynomial::Polynomial;
use super::trace::TraceHandle;
use crate::arithmetic::register::{CellType, DataRegister, U16Array};
use crate::arithmetic::Register;

pub const LIMB: u32 = 2u32.pow(16);

pub trait FieldParameters<const N_LIMBS: usize>: Send + Sync + Copy + 'static {
    const MODULUS: [u16; N_LIMBS];
    const WITNESS_OFFSET: usize;

    fn modulus_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::MODULUS.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceHandle<F, D> {
    pub fn write_field<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize>(
        &self,
        row_index: usize,
        a_int: &BigUint,
        a: FieldRegister<P, N_LIMBS>,
    ) -> Result<()> {
        let p_a = Polynomial::<F>::from_biguint_field(a_int, 16, 16);
        self.write_data(row_index, a, p_a.into_vec())
    }
}

pub fn modulus_field_iter<F: Field, P: FieldParameters<N_LIMBS>, const N_LIMBS: usize>(
) -> impl Iterator<Item = F> {
    P::MODULUS.into_iter().map(|x| F::from_canonical_u16(x))
}

#[derive(Debug, Clone, Copy)]
pub struct FieldRegister<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> {
    array: U16Array<N_LIMBS>,
    _marker: core::marker::PhantomData<P>,
}

impl<const N_LIMBS: usize, P: FieldParameters<N_LIMBS>> DataRegister for FieldRegister<P, N_LIMBS> {
    const CELL: Option<CellType> = Some(CellType::U16);

    fn from_raw_register(register: Register) -> Self {
        Self {
            array: U16Array::from_raw_register(register),
            _marker: core::marker::PhantomData,
        }
    }

    fn register(&self) -> &Register {
        self.array.register()
    }

    fn size_of() -> usize {
        N_LIMBS
    }

    fn into_raw_register(self) -> Register {
        self.array.into_raw_register()
    }
}

/// The parameters for the Fp25519 field of modulues 2^255-19.
#[derive(Debug, Clone, Copy)]
pub struct Fp25519Param;

pub type Fp25519 = FieldRegister<Fp25519Param, 16>;

impl FieldParameters<16> for Fp25519Param {
    const MODULUS: [u16; 16] = [
        65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535, 32767,
    ];

    const WITNESS_OFFSET: usize = 1usize << 20;

    fn modulus_biguint() -> BigUint {
        (BigUint::one() << 255) - BigUint::from(19u32)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FpInstruction<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> {
    Add(FpAdd<P, N_LIMBS>),
    Mul(FpMul<P, N_LIMBS>),
    Quad(FpQuad<P, N_LIMBS>),
}

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> From<FpAdd<P, N_LIMBS>>
    for FpInstruction<P, N_LIMBS>
{
    fn from(add: FpAdd<P, N_LIMBS>) -> Self {
        Self::Add(add)
    }
}

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> From<FpMul<P, N_LIMBS>>
    for FpInstruction<P, N_LIMBS>
{
    fn from(mul: FpMul<P, N_LIMBS>) -> Self {
        Self::Mul(mul)
    }
}

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> From<FpQuad<P, N_LIMBS>>
    for FpInstruction<P, N_LIMBS>
{
    fn from(quad: FpQuad<P, N_LIMBS>) -> Self {
        Self::Quad(quad)
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const N: usize, FP: FieldParameters<N>>
    Instruction<F, D> for FpInstruction<FP, N>
{
    fn memory_vec(&self) -> Vec<Register> {
        match self {
            FpInstruction::Add(add) => <FpAdd<FP, N> as Instruction<F, D>>::memory_vec(add),
            FpInstruction::Mul(mul) => <FpMul<FP, N> as Instruction<F, D>>::memory_vec(mul),
            FpInstruction::Quad(quad) => <FpQuad<FP, N> as Instruction<F, D>>::memory_vec(quad),
        }
    }

    fn witness_data(&self) -> Option<super::register::WitnessData> {
        match self {
            FpInstruction::Add(add) => <FpAdd<FP, N> as Instruction<F, D>>::witness_data(add),
            FpInstruction::Mul(mul) => <FpMul<FP, N> as Instruction<F, D>>::witness_data(mul),
            FpInstruction::Quad(quad) => <FpQuad<FP, N> as Instruction<F, D>>::witness_data(quad),
        }
    }

    fn set_witness(&mut self, witness: Register) -> Result<()> {
        match self {
            FpInstruction::Add(add) => {
                <FpAdd<FP, N> as Instruction<F, D>>::set_witness(add, witness)
            }
            FpInstruction::Mul(mul) => {
                <FpMul<FP, N> as Instruction<F, D>>::set_witness(mul, witness)
            }
            FpInstruction::Quad(quad) => {
                <FpQuad<FP, N> as Instruction<F, D>>::set_witness(quad, witness)
            }
        }
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            FpInstruction::Add(add) => {
                <FpAdd<FP, N> as Instruction<F, D>>::assign_row(add, trace_rows, row, row_index)
            }
            FpInstruction::Mul(mul) => {
                <FpMul<FP, N> as Instruction<F, D>>::assign_row(mul, trace_rows, row, row_index)
            }
            FpInstruction::Quad(quad) => {
                <FpQuad<FP, N> as Instruction<F, D>>::assign_row(quad, trace_rows, row, row_index)
            }
        }
    }

    fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: crate::vars::StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        P: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        match self {
            FpInstruction::Add(add) => {
                <FpAdd<FP, N> as Instruction<F, D>>::packed_generic_constraints(
                    add,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Mul(mul) => {
                <FpMul<FP, N> as Instruction<F, D>>::packed_generic_constraints(
                    mul,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Quad(quad) => {
                <FpQuad<FP, N> as Instruction<F, D>>::packed_generic_constraints(
                    quad,
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
                <FpAdd<FP, N> as Instruction<F, D>>::ext_circuit_constraints(
                    add,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Mul(mul) => {
                <FpMul<FP, N> as Instruction<F, D>>::ext_circuit_constraints(
                    mul,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            FpInstruction::Quad(quad) => {
                <FpQuad<FP, N> as Instruction<F, D>>::ext_circuit_constraints(
                    quad,
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
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
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
        const NUM_ARITHMETIC_COLUMNS: usize = FpQuad::<Fp25519Param, 16>::num_quad_columns();
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = FpInstruction<Fp25519Param, 16>;
    }

    #[test]
    fn test_fpquad_instructions() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<FpInstructionTest, F, D>;

        // build the stark
        let mut builder = ChipBuilder::<FpInstructionTest, F, D>::new();

        let a = builder.alloc_local::<Fp>().unwrap();
        let b = builder.alloc_local::<Fp>().unwrap();
        let c = builder.alloc_local::<Fp>().unwrap();
        let d = builder.alloc_local::<Fp>().unwrap();
        let result = builder.alloc_local::<Fp>().unwrap();

        let quad = builder.fpquad(&a, &b, &c, &d, &result).unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();
        builder.write_data(&c).unwrap();
        builder.write_data(&d).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus_biguint();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let c_int = rng.gen_biguint(256) % &p;
            let d_int = rng.gen_biguint(256) % &p;
            let handle = handle.clone();
            rayon::spawn(move || {
                handle.write_field(i as usize, &a_int, a).unwrap();
                handle.write_field(i as usize, &b_int, b).unwrap();
                handle.write_field(i as usize, &c_int, c).unwrap();
                handle.write_field(i as usize, &d_int, d).unwrap();
                handle
                    .write_fpquad(i as usize, &a_int, &b_int, &c_int, &d_int, quad)
                    .unwrap();
            });
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
