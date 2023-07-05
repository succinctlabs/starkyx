//! A vartation of Starky that includes random verifier challenges (RAIR)
//!
//!

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::fri::structure::{
    FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOracleInfo,
    FriPolynomialInfo,
};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::GenericConfig;

use self::config::StarkyConfig;
use super::parser::{RecursiveStarkParser, StarkParser};
use crate::air::parser::AirParser;
use crate::air::RAir;
use crate::stark::Stark;

pub mod config;
pub mod gadget;
pub mod generator;
pub mod proof;
pub mod prover;
pub mod verifier;

pub trait Plonky2Stark<F: RichField + Extendable<D>, const D: usize>: Sync {
    const COLUMNS: usize;

    type Air;

    fn air(&self) -> &Self::Air;

    fn num_quotient_polys<AP: AirParser, C: GenericConfig<D, F = F>>(
        &self,
        config: &StarkyConfig<F, C, D>,
    ) -> usize
    where
        Self::Air: RAir<AP>,
    {
        self.air().quotient_degree_factor() * config.num_challenges
    }

    /// Computes the FRI instance used to prove this Stark.
    fn fri_instance<AP: AirParser, C: GenericConfig<D, F = F>>(
        &self,
        zeta: F::Extension,
        g: F,
        config: &StarkyConfig<F, C, D>,
    ) -> FriInstanceInfo<F, D>
    where
        Self::Air: RAir<AP>,
    {
        let mut oracles = vec![];
        let mut trace_info: Vec<FriPolynomialInfo> = vec![];

        for length in self.air().round_lengths() {
            let round_info = FriPolynomialInfo::from_range(oracles.len(), 0..length);
            trace_info.extend(round_info);
            oracles.push(FriOracleInfo {
                num_polys: length,
                blinding: false,
            });
        }

        let num_quotient_polys = self.air().quotient_degree_factor() * config.num_challenges;
        let quotient_info = FriPolynomialInfo::from_range(oracles.len(), 0..num_quotient_polys);
        oracles.push(FriOracleInfo {
            num_polys: num_quotient_polys,
            blinding: false,
        });

        let zeta_batch = FriBatchInfo {
            point: zeta,
            polynomials: [trace_info.clone(), quotient_info].concat(),
        };
        let zeta_next_batch = FriBatchInfo {
            point: zeta.scalar_mul(g),
            polynomials: trace_info,
        };

        let batches = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfo { oracles, batches }
    }

    /// Computes the FRI instance used to prove this Stark.
    fn fri_instance_target<AP: AirParser, C: GenericConfig<D, F = F>>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
        g: F,
        config: &StarkyConfig<F, C, D>,
    ) -> FriInstanceInfoTarget<D>
    where
        Self::Air: RAir<AP>,
    {
        let mut oracles = vec![];
        let mut trace_info: Vec<FriPolynomialInfo> = vec![];

        for length in self.air().round_lengths() {
            let round_info = FriPolynomialInfo::from_range(oracles.len(), 0..length);
            trace_info.extend(round_info);
            oracles.push(FriOracleInfo {
                num_polys: length,
                blinding: false,
            });
        }

        let num_quotient_polys = self.air().quotient_degree_factor() * config.num_challenges;
        let quotient_info = FriPolynomialInfo::from_range(oracles.len(), 0..num_quotient_polys);
        oracles.push(FriOracleInfo {
            num_polys: num_quotient_polys,
            blinding: false,
        });

        let zeta_batch = FriBatchInfoTarget {
            point: zeta,
            polynomials: [trace_info.clone(), quotient_info].concat(),
        };

        let zeta_next = builder.mul_const_extension(g, zeta);
        let zeta_next_batch = FriBatchInfoTarget {
            point: zeta_next,
            polynomials: trace_info,
        };

        let batches = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfoTarget { oracles, batches }
    }
}

impl<'a, T, F, C: GenericConfig<D, F = F>, FE, P, const D: usize, const D2: usize>
    Stark<StarkParser<'a, F, FE, P, D, D2>, StarkyConfig<F, C, D>> for T
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    T: Plonky2Stark<F, D>,
    T::Air: RAir<StarkParser<'a, F, FE, P, D, D2>>,
{
    type Air = T::Air;

    fn air(&self) -> &Self::Air {
        self.air()
    }
}

impl<'a, T, F, C: GenericConfig<D, F = F>, const D: usize>
    Stark<RecursiveStarkParser<'a, F, D>, StarkyConfig<F, C, D>> for T
where
    F: RichField + Extendable<D>,
    T: Plonky2Stark<F, D>,
    T::Air: RAir<RecursiveStarkParser<'a, F, D>>,
{
    type Air = T::Air;

    fn air(&self) -> &Self::Air {
        self.air()
    }
}

#[derive(Debug, Clone)]
pub struct Starky<A, const COLUMNS: usize> {
    air: A,
}

impl<A, const COLUMNS: usize> Starky<A, COLUMNS> {
    pub fn new(air: A) -> Self {
        Self { air }
    }
}

impl<F, A: Sync, const D: usize, const COLUMNS: usize> Plonky2Stark<F, D> for Starky<A, COLUMNS>
where
    F: RichField + Extendable<D>,
{
    const COLUMNS: usize = COLUMNS;
    type Air = A;

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::packable::Packable;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use plonky2::util::timing::TimingTree;
    use prover::StarkyProver;
    use verifier::StarkyVerifier;

    use super::*;
    use crate::air::fibonacci::FibonacciAir;
    use crate::math::prelude::*;
    use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    use crate::plonky2::stark::gadget::StarkGadget;
    use crate::plonky2::stark::verifier::set_stark_proof_target;
    use crate::trace::generator::ConstantGenerator;

    #[test]
    fn test_plonky2_fibonacci_stark() {
        type F = GoldilocksField;
        type C = PoseidonGoldilocksConfig;
        type SC = PoseidonGoldilocksStarkConfig;
        const D: usize = 2;

        let num_rows = 1 << 5 as usize;
        let air = FibonacciAir::new();
        let stark = Starky::<FibonacciAir, 4>::new(air);

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(num_rows - 1, F::ZERO, F::ONE),
        ];

        let trace = FibonacciAir::generate_trace(F::ZERO, F::ONE, num_rows);
        let trace_generator = ConstantGenerator::new(trace);

        let config = SC::standard_fast_config(num_rows);

        let proof = StarkyProver::<F, C, F, <F as Packable>::Packing, 2, 1>::prove(
            &config,
            &stark,
            &trace_generator,
            &public_inputs,
        )
        .unwrap();

        // Verify the proof as a stark
        StarkyVerifier::verify(&config, &stark, proof.clone(), &public_inputs).unwrap();

        // Test the recursive proof.
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);
        let virtual_proof = recursive_builder.add_virtual_stark_proof(&stark, &config);

        recursive_builder.print_gate_counts(0);
        let mut rec_pw = PartialWitness::new();
        // Set public inputs.
        let public_input_targets = recursive_builder.add_virtual_targets(3);
        for (&pi_t, pi) in public_input_targets.iter().zip(public_inputs) {
            rec_pw.set_target(pi_t, pi);
        }
        set_stark_proof_target(&mut rec_pw, &virtual_proof, &proof);
        recursive_builder.verify_stark_proof(&config, &stark, virtual_proof, &public_input_targets);
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
