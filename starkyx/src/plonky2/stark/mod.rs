//! A vartation of Starky that includes random verifier challenges (RAIR)
//!
//!

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::fri::structure::{
    FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOracleInfo,
    FriPolynomialInfo,
};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use serde::{Deserialize, Serialize};

use self::config::{CurtaConfig, StarkyConfig};
use crate::air::RAirData;

pub mod config;
pub mod gadget;
pub mod generator;
pub mod proof;
pub mod prover;
pub mod verifier;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Starky<A> {
    pub air: A,
}

impl<A> Starky<A> {
    pub fn new(air: A) -> Self {
        Self { air }
    }
}

impl<A> Starky<A> {
    pub fn air(&self) -> &A {
        &self.air
    }

    pub fn num_quotient_polys<
        F: RichField + Extendable<D>,
        C: CurtaConfig<D, F = F>,
        const D: usize,
    >(
        &self,
        config: &StarkyConfig<C, D>,
    ) -> usize
    where
        A: RAirData,
    {
        self.air().quotient_degree_factor() * config.num_challenges
    }

    /// Computes the FRI instance used to prove this Stark.
    pub fn fri_instance<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize>(
        &self,
        zeta: F::Extension,
        g: F,
        config: &StarkyConfig<C, D>,
    ) -> FriInstanceInfo<F, D>
    where
        A: RAirData,
    {
        let mut oracles = vec![];
        let mut trace_info: Vec<FriPolynomialInfo> = vec![];

        for round in self.air().round_data() {
            let length = round.num_columns;
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
    pub fn fri_instance_target<
        C: CurtaConfig<D, F = F>,
        F: RichField + Extendable<D>,
        const D: usize,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
        g: F,
        config: &StarkyConfig<C, D>,
    ) -> FriInstanceInfoTarget<D>
    where
        A: RAirData,
    {
        let mut oracles = vec![];
        let mut trace_info: Vec<FriPolynomialInfo> = vec![];

        for round in self.air().round_data() {
            let length = round.num_columns;
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

#[cfg(test)]
pub(crate) mod tests {
    use core::fmt::Debug;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::AlgebraicHasher;
    use plonky2::util::timing::TimingTree;
    use serde::de::DeserializeOwned;

    use super::generator::simple::SimpleStarkWitnessGenerator;
    use super::*;
    use crate::air::fibonacci::FibonacciAir;
    use crate::chip::builder::tests::ArithmeticGenerator;
    use crate::chip::{AirParameters, Chip};
    use crate::math::prelude::*;
    use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    use crate::plonky2::stark::gadget::StarkGadget;
    use crate::plonky2::stark::prover::StarkyProver;
    use crate::plonky2::stark::verifier::StarkyVerifier;
    use crate::plonky2::{Plonky2Air, StarkyAir};
    use crate::trace::generator::{ConstantGenerator, TraceGenerator};

    /// Generate the proof and verify as a stark
    pub(crate) fn test_starky<
        A,
        T,
        F: RichField + Extendable<D>,
        C: CurtaConfig<D, F = F, FE = F::Extension>,
        const D: usize,
    >(
        stark: &Starky<A>,
        config: &StarkyConfig<C, D>,
        trace_generator: &T,
        public_inputs: &[F],
    ) where
        A: 'static + Debug + Send + Sync,
        A: StarkyAir<F, D>,
        T: TraceGenerator<F, A>,
        T::Error: Into<anyhow::Error>,
    {
        let proof =
            StarkyProver::<F, C, D>::prove(config, stark, trace_generator, public_inputs).unwrap();

        // Verify the proof as a stark
        StarkyVerifier::verify(config, stark, proof, public_inputs).unwrap();
    }

    /// Generate a Stark proof and a recursive proof using the witness generator
    pub(crate) fn test_recursive_starky<
        L: AirParameters<Field = F>,
        F: RichField + Extendable<D>,
        C: CurtaConfig<D, F = F, FE = F::Extension> + 'static + Serialize + DeserializeOwned,
        const D: usize,
    >(
        stark: Starky<Chip<L>>,
        config: StarkyConfig<C, D>,
        trace_generator: ArithmeticGenerator<L>,
        public_inputs: &[F],
    ) where
        C::Hasher: AlgebraicHasher<F>,
        Chip<L>: Plonky2Air<F, D>,
    {
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config_rec);
        let virtual_proof = builder.add_virtual_stark_proof(&stark, &config);

        builder.print_gate_counts(0);
        let mut pw = PartialWitness::new();
        // Set public inputs.
        let public_input_targets = builder.add_virtual_targets(public_inputs.len());
        for (&pi_t, &pi) in public_input_targets.iter().zip(public_inputs.iter()) {
            pw.set_target(pi_t, pi);
        }
        builder.verify_stark_proof(&config, &stark, &virtual_proof, &public_input_targets);

        let generator = SimpleStarkWitnessGenerator::new(
            config,
            stark,
            virtual_proof,
            public_input_targets,
            trace_generator,
        );
        builder.add_simple_generator(generator);

        let data = builder.build::<C::GenericConfig>();
        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof =
            plonky2::plonk::prover::prove(&data.prover_only, &data.common, pw, &mut timing)
                .unwrap();
        timing.print();
        data.verify(recursive_proof).unwrap();
    }

    #[test]
    fn test_plonky2_fibonacci_stark() {
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let num_rows = 1 << 5usize;
        let air = FibonacciAir::new();
        let stark = Starky::<FibonacciAir>::new(air);

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(num_rows - 1, F::ZERO, F::ONE),
        ];

        let trace = FibonacciAir::generate_trace(F::ZERO, F::ONE, num_rows);
        let trace_generator = ConstantGenerator::new(trace);

        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &trace_generator, &public_inputs);
    }
}
