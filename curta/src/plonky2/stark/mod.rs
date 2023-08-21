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
use crate::air::{RAir, RAirData};
use crate::stark::Stark;

pub mod config;
pub mod gadget;
pub mod generator;
pub mod proof;
pub mod prover;
pub mod verifier;

#[derive(Debug, Clone)]
pub struct Starky<A> {
    pub air: A,
}

impl<A> Starky<A> {
    pub fn new(air: A) -> Self {
        Self { air }
    }
}

impl<A> Starky<A> {
    fn air(&self) -> &A {
        &self.air
    }

    fn num_quotient_polys<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &self,
        config: &StarkyConfig<F, C, D>,
    ) -> usize
    where
        A: RAirData,
    {
        self.air().quotient_degree_factor() * config.num_challenges
    }

    /// Computes the FRI instance used to prove this Stark.
    fn fri_instance<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        &self,
        zeta: F::Extension,
        g: F,
        config: &StarkyConfig<F, C, D>,
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
    fn fri_instance_target<
        C: GenericConfig<D, F = F>,
        F: RichField + Extendable<D>,
        const D: usize,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
        g: F,
        config: &StarkyConfig<F, C, D>,
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

impl<'a, A, F, C: GenericConfig<D, F = F>, FE, P, const D: usize, const D2: usize>
    Stark<StarkParser<'a, F, FE, P, D, D2>, StarkyConfig<F, C, D>> for Starky<A>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    A: RAir<StarkParser<'a, F, FE, P, D, D2>>,
{
    type Air = A;

    fn air(&self) -> &Self::Air {
        self.air()
    }
}

impl<'a, A, F, C: GenericConfig<D, F = F>, const D: usize>
    Stark<RecursiveStarkParser<'a, F, D>, StarkyConfig<F, C, D>> for Starky<A>
where
    F: RichField + Extendable<D>,
    A: RAir<RecursiveStarkParser<'a, F, D>>,
{
    type Air = A;

    fn air(&self) -> &Self::Air {
        self.air()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use core::fmt::Debug;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::packable::Packable;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::AlgebraicHasher;
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::air::fibonacci::FibonacciAir;
    use crate::math::prelude::*;
    use crate::plonky2::parser::global::{GlobalRecursiveStarkParser, GlobalStarkParser};
    use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    use crate::plonky2::stark::gadget::StarkGadget;
    use crate::plonky2::stark::generator::simple::SimpleStarkWitnessGenerator;
    use crate::plonky2::stark::prover::StarkyProver;
    use crate::plonky2::stark::verifier::StarkyVerifier;
    use crate::trace::generator::{ConstantGenerator, TraceGenerator};

    /// Generate the proof and verify as a stark
    pub(crate) fn test_starky<
        A: 'static + Debug + Send + Sync,
        T,
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F, FE = F::Extension>,
        const D: usize,
    >(
        stark: &Starky<A>,
        config: &StarkyConfig<F, C, D>,
        trace_generator: &T,
        public_inputs: &[F],
    ) where
        A: for<'a> RAir<StarkParser<'a, F, C::FE, C::FE, D, D>>
            + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>
            + for<'a> RAir<GlobalStarkParser<'a, F, F, F, D, 1>>,
        T: TraceGenerator<F, A>,
        T::Error: Into<anyhow::Error>,
    {
        let proof = StarkyProver::<F, C, F, <F as Packable>::Packing, D, 1>::prove(
            config,
            stark,
            trace_generator,
            public_inputs,
        )
        .unwrap();

        // Verify the proof as a stark
        StarkyVerifier::verify(config, stark, proof, public_inputs).unwrap();
    }

    /// Generate a Stark proof and a recursive proof using the witness generator
    pub(crate) fn test_recursive_starky<
        A: 'static + Debug + Send + Sync,
        T,
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F, FE = F::Extension> + 'static,
        const D: usize,
    >(
        stark: Starky<A>,
        config: StarkyConfig<F, C, D>,
        trace_generator: T,
        public_inputs: &[F],
    ) where
        C::Hasher: AlgebraicHasher<F>,
        A: for<'a> RAir<RecursiveStarkParser<'a, F, D>>
            + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>
            + for<'a> RAir<GlobalRecursiveStarkParser<'a, F, D>>,
        T: Clone + Debug + Send + Sync + 'static + TraceGenerator<F, A>,
        T::Error: Into<anyhow::Error>,
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
        builder.verify_stark_proof(
            &config,
            &stark,
            virtual_proof.clone(),
            &public_input_targets,
        );

        let generator = SimpleStarkWitnessGenerator::new(
            config,
            stark,
            virtual_proof,
            public_input_targets,
            trace_generator,
        );
        builder.add_simple_generator(generator);

        let data = builder.build::<C>();
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

        // Test the recursive proof.
        test_recursive_starky(stark, config, trace_generator, &public_inputs);
    }
}
