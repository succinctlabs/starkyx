use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::instruction::Instruction;
use super::register::{Register, WitnessData};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConstraintBool(pub Register);

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for ConstraintBool {
    fn witness_data(&self) -> Option<WitnessData> {
        None
    }

    fn memory_vec(&self) -> Vec<Register> {
        vec![self.0]
    }

    fn set_witness(&mut self, _witness: Register) -> Result<()> {
        Ok(())
    }

    fn assign_row(&self, _trace_rows: &mut [Vec<F>], _row: &mut [F], _row_index: usize) {}

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
        let values = self.0.packed_entries_slice(&vars);
        for value in values.iter() {
            yield_constr.constraint(value.square() - *value);
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let values = self.0.evaluation_targets(&vars);
        for value in values.iter() {
            let square = builder.square_extension(*value);
            let constraint = builder.sub_extension(square, *value);
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

    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::register::{BitRegister, U16Array};
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Debug, Clone, Copy)]
    pub struct BoolTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for BoolTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 1;
        const NUM_FREE_COLUMNS: usize = 2;

        type Instruction = super::ConstraintBool;
    }

    #[test]
    fn test_bool_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<BoolTest, F, D>;

        let mut builder = ChipBuilder::<BoolTest, F, D>::new();

        let bit_one = builder.alloc_local::<BitRegister>().unwrap();
        builder.write_data(&bit_one).unwrap();
        let bit_zero = builder.alloc_local::<BitRegister>().unwrap();
        builder.write_data(&bit_zero).unwrap();

        let dummy = builder.alloc_local::<U16Array<1>>().unwrap();
        builder.write_data(&dummy).unwrap();

        let (chip, spec) = builder.build();

        // Test successful proof
        // Construct the trace
        let num_rows = 2u64.pow(5) as usize;
        let (handle, generator) = trace::<F, D>(spec.clone());
        for i in 0..num_rows {
            handle.write_data(i, bit_one, vec![F::ONE]).unwrap();
            handle.write_data(i, bit_zero, vec![F::ZERO]).unwrap();
            handle.write_data(i, dummy, vec![F::ZERO; 1]).unwrap();
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows as usize).unwrap();

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
        let (handle, generator) = crate::arithmetic::trace::trace::<F, D>(spec);
        for i in 0..num_rows {
            handle.write_data(i, bit_zero, vec![F::ZERO]).unwrap();
            handle
                .write_data(i, bit_one, vec![F::ONE + F::ONE])
                .unwrap();
            handle.write_data(i, dummy, vec![F::ZERO; 1]).unwrap();
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

        let res = verify_stark_proof(stark.clone(), proof.clone(), &config);
        assert!(res.is_err())
    }
}
