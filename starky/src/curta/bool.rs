use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::builder::StarkBuilder;
use super::chip::StarkParameters;
use super::instruction::Instruction;
use super::register::{BitRegister, MemorySlice, Register};
use super::trace::TraceWriter;
use crate::curta::register::RegisterSerializable;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct Selector<T> {
    bit: BitRegister,
    true_value: T,
    false_value: T,
    result: T,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub fn selector<T: Register>(
        &mut self,
        bit: &BitRegister,
        a: &T,
        b: &T,
        result: &T,
    ) -> Result<Selector<T>>
    where
        L::Instruction: From<Selector<T>>,
    {
        let instr = Selector::new(*bit, *a, *b, *result);
        self.insert_instruction(instr.clone().into())?;
        Ok(instr)
    }
}

impl<T> Selector<T> {
    pub fn new(bit: BitRegister, true_value: T, false_value: T, result: T) -> Self {
        Self {
            bit,
            true_value,
            false_value,
            result,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    #[allow(dead_code)]
    pub fn write_bit(&self, row_index: usize, bit: bool, data: &BitRegister) -> Result<()> {
        self.write_data(row_index, *data, vec![F::from_canonical_u16(bit as u16)])
    }
}

impl<F: RichField + Extendable<D>, const D: usize, T: Register> Instruction<F, D> for Selector<T> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![*self.result.register()]
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.result.register().assign(trace_rows, 0, row, row_index);
    }

    fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let bit_slice = self.bit.register().packed_generic_vars(&vars);
        let true_value = self.true_value.register().packed_generic_vars(&vars);
        let false_value = self.false_value.register().packed_generic_vars(&vars);
        let result = self.result.register().packed_generic_vars(&vars);

        debug_assert!(bit_slice.len() == 1);
        let bit = bit_slice[0];
        for ((x_true, x_false), x) in true_value.iter().zip(false_value.iter()).zip(result.iter()) {
            yield_constr.constraint(*x_true * bit + *x_false * (P::from(FE::ONE) - bit) - *x);
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let bit_slice = self.bit.register().ext_circuit_vars(&vars);
        let true_value = self.true_value.register().ext_circuit_vars(&vars);
        let false_value = self.false_value.register().ext_circuit_vars(&vars);
        let result = self.result.register().ext_circuit_vars(&vars);

        debug_assert!(bit_slice.len() == 1);
        let bit = bit_slice[0];
        let one = builder.constant_extension(F::Extension::ONE);
        let one_minus_bit = builder.sub_extension(one, bit);
        for ((x_true, x_false), x) in true_value.iter().zip(false_value.iter()).zip(result.iter()) {
            let bit_x_true = builder.mul_extension(*x_true, bit);
            let one_minus_bit_x_false = builder.mul_extension(*x_false, one_minus_bit);
            let expected_res = builder.add_extension(bit_x_true, one_minus_bit_x_false);
            let constraint = builder.sub_extension(expected_res, *x);
            yield_constr.constraint(builder, constraint);
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::extension::Extendable;
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::RichField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{StarkParameters, TestStark};
    use crate::curta::register::BitRegister;
    use crate::curta::trace::trace;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Debug, Clone, Copy)]
    pub struct BoolTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for BoolTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 10;
        const NUM_FREE_COLUMNS: usize = 10;
        type Instruction = Selector<BitRegister>;
    }

    #[test]
    fn test_bool_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<BoolTest, F, D>;

        let mut builder = StarkBuilder::<BoolTest, F, D>::new();

        let bit_one = builder.alloc::<BitRegister>();
        builder.write_data(&bit_one).unwrap();
        let bit_zero = builder.alloc::<BitRegister>();
        builder.write_data(&bit_zero).unwrap();

        let dummy = builder.alloc::<BitRegister>();
        builder.write_data(&dummy).unwrap();

        let (chip, spec) = builder.build();

        // Test successful proof
        // Construct the trace
        let num_rows = 2u64.pow(5) as usize;
        let (handle, generator) = trace::<F, D>(spec.clone());
        for i in 0..num_rows {
            handle.write_data(i, bit_one, vec![F::ONE]).unwrap();
            handle.write_data(i, bit_zero, vec![F::ZERO]).unwrap();
            handle.write_data(i, dummy, vec![F::ZERO]).unwrap();
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows).unwrap();

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip.clone());

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

        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut TimingTree::default(),
        )
        .unwrap();

        recursive_data.verify(recursive_proof).unwrap();

        // test unsuccesfull proof
        // Construct the trace
        let num_rows = 2u64.pow(5) as usize;
        let (handle, generator) = crate::curta::trace::trace::<F, D>(spec);
        for i in 0..num_rows {
            handle.write_data(i, bit_zero, vec![F::ZERO]).unwrap();
            handle
                .write_data(i, bit_one, vec![F::ONE + F::ONE])
                .unwrap();
            handle.write_data(i, dummy, vec![F::ZERO; 1]).unwrap();
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

        let res = verify_stark_proof(stark, proof, &config);
        assert!(res.is_err())
    }

    #[test]
    fn test_selector() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<BoolTest, F, D>;

        let mut builder = StarkBuilder::<BoolTest, F, D>::new();

        let bit = builder.alloc::<BitRegister>();
        builder.write_data(&bit).unwrap();

        let x = builder.alloc::<BitRegister>();
        builder.write_data(&x).unwrap();
        let y = builder.alloc::<BitRegister>();
        builder.write_data(&y).unwrap();
        let result = builder.alloc::<BitRegister>();

        let sel = builder.selector(&bit, &x, &y, &result).unwrap();

        let (chip, spec) = builder.build();

        // Test successful proof
        // Construct the trace
        let num_rows = 2u64.pow(5) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        for i in 0..num_rows {
            let x_i = 0u16;
            let y_i = 1u16;
            let bit_i = if i % 2 == 0 { true } else { false };
            handle.write_bit(i, bit_i, &bit).unwrap();
            let res = if i % 2 == 0 { x_i } else { y_i };
            handle
                .write_data(i, x, vec![F::from_canonical_u16(x_i)])
                .unwrap();
            handle
                .write_data(i, y, vec![F::from_canonical_u16(y_i)])
                .unwrap();
            handle
                .write(i, sel, vec![F::from_canonical_u16(res)])
                .unwrap();
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

        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut TimingTree::default(),
        )
        .unwrap();

        recursive_data.verify(recursive_proof).unwrap();
    }
}
