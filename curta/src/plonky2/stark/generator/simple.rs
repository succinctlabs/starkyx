use core::fmt::Debug;

use plonky2::field::extension::Extendable;
use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::util::serialization::IoResult;

use super::super::config::StarkyConfig;
use super::super::proof::StarkProofTarget;
use super::super::prover::StarkyProver;
use super::super::verifier::set_stark_proof_target;
use super::super::Plonky2Stark;
use crate::air::RAir;
use crate::plonky2::parser::StarkParser;
use crate::trace::generator::TraceGenerator;

#[derive(Debug, Clone)]
pub struct SimpleStarkWitnessGenerator<S, T, F, C, P, const D: usize> {
    config: StarkyConfig<F, C, D>,
    pub stark: S,
    pub proof_target: StarkProofTarget<D>,
    pub public_input_targets: Vec<Target>,
    pub trace_generator: T,
    _marker: core::marker::PhantomData<P>,
}

impl<S, T, F: RichField, C, const D: usize>
    SimpleStarkWitnessGenerator<S, T, F, C, <F as Packable>::Packing, D>
{
    pub fn new(
        config: StarkyConfig<F, C, D>,
        stark: S,
        proof_target: StarkProofTarget<D>,
        public_input_targets: Vec<Target>,
        trace_generator: T,
    ) -> Self {
        Self {
            config,
            stark,
            proof_target,
            public_input_targets,
            trace_generator,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S, T, F, C, P, const D: usize> SimpleGenerator<F, D>
    for SimpleStarkWitnessGenerator<S, T, F, C, P, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    C::Hasher: AlgebraicHasher<F>,
    P: PackedField<Scalar = F>,
    S: Plonky2Stark<F, D> + Debug + Send + 'static,
    S::Air: for<'a> RAir<StarkParser<'a, F, F, P, D, 1>>,
    T: Debug + Send + Sync + 'static + TraceGenerator<F, S::Air>,
    T::Error: Into<anyhow::Error>,
    [(); S::COLUMNS]:,
{
    fn id(&self) -> String {
        "SimpleStarkWitnessGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.public_input_targets.clone()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let public_inputs = witness.get_targets(&self.public_input_targets);

        let proof = StarkyProver::<F, C, F, P, D, 1>::prove(
            &self.config,
            &self.stark,
            &self.trace_generator,
            &public_inputs,
        )
        .unwrap();

        set_stark_proof_target(out_buffer, &self.proof_target, &proof);
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        unimplemented!("SimpleStarkWitnessGenerator::serialize")
    }

    fn deserialize(
        _src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Self>
    where
        Self: Sized,
    {
        unimplemented!("SimpleStarkWitnessGenerator::deserialize")
    }
}
