use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use super::Stark;
use crate::curta::air::parser::AirParser;
use crate::curta::trace::types::ConstantGenerator;

#[derive(Debug, Clone)]
pub struct FibonacciStark<F, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> FibonacciStark<F, D> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    pub fn generator(&self, x0: F, x1: F, num_rows: usize) -> ConstantGenerator<F> {
        let mut trace_rows = (0..num_rows)
            .scan([x0, x1, F::ZERO, F::ONE], |acc, _| {
                let tmp = *acc;
                acc[0] = tmp[1];
                acc[1] = tmp[0] + tmp[1];
                acc[2] = tmp[2] + F::ONE;
                acc[3] = tmp[3] + F::ONE;
                Some(tmp)
            })
            .map(|arr| arr.to_vec())
            .collect::<Vec<_>>();
        trace_rows[num_rows - 1][3] = F::ZERO; // So that column 2 and 3 are permutation of one another.
        ConstantGenerator::from_rows(trace_rows)
    }
}

pub fn fibonacci<F: Field>(n: usize, x0: F, x1: F) -> F {
    (0..n).fold((x0, x1), |x, _| (x.1, x.0 + x.1)).1
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D, 1> for FibonacciStark<F, D> {
    const COLUMNS: usize = 4;
    const PUBLIC_INPUTS: usize = 3;
    const CHALLENGES: usize = 0;

    /// Columns for each round
    fn round_lengths(&self) -> [usize; 1] {
        [4]
    }

    /// The number of challenges per round
    fn num_challenges(&self, _round: usize) -> usize {
        0
    }

    fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) {
        // Check public inputs.
        let pis_constraints = [
            parser.sub(parser.local_slice()[0], parser.public_slice()[0]),
            parser.sub(parser.local_slice()[1], parser.public_slice()[1]),
            parser.sub(parser.local_slice()[1], parser.public_slice()[2]),
        ];
        parser.constraint_first_row(pis_constraints[0]);
        parser.constraint_first_row(pis_constraints[1]);
        parser.constraint_last_row(pis_constraints[2]);

        // x0' <- x1
        let first_col_constraint = parser.sub(parser.next_slice()[0], parser.local_slice()[1]);
        parser.constraint_transition(first_col_constraint);
        // x1' <- x0 + x1
        let second_col_constraint = {
            let tmp = parser.sub(parser.next_slice()[1], parser.local_slice()[0]);
            parser.sub(tmp, parser.local_slice()[1])
        };
        parser.constraint_transition(second_col_constraint);
    }

    fn constraint_degree(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::stark::prover::prove;
    use crate::curta::stark::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::curta::stark::verifier::verify_stark_proof;
    use crate::curta::trace::types::ConstantGenerator;

    #[test]
    fn test_new_fibonacci_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = FibonacciStark<F, D>;
        type T = ConstantGenerator<F>;

        let config = StarkConfig::standard_fast_config();
        let num_rows = 1 << 5;
        let public_inputs = [F::ZERO, F::ONE, fibonacci(num_rows - 1, F::ZERO, F::ONE)];
        let stark = S::new();

        let trace_generator = stark.generator(public_inputs[0], public_inputs[1], num_rows);
        let proof = prove::<F, C, S, T, D, 1>(
            stark.clone(),
            &config,
            trace_generator,
            num_rows,
            public_inputs,
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
        verify_stark_proof_circuit::<F, C, S, D, 1>(
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
