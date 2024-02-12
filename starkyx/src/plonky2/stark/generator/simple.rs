use core::fmt::Debug;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult};
use serde::{Deserialize, Serialize};

use super::super::config::StarkyConfig;
use super::super::proof::StarkProofTarget;
use super::super::prover::StarkyProver;
use super::super::verifier::set_stark_proof_target;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::{AirParameters, Chip};
use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::Starky;
use crate::plonky2::Plonky2Air;
use crate::utils::serde::{BufferRead, BufferWrite};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SimpleStarkWitnessGenerator<L: AirParameters, C, const D: usize> {
    pub config: StarkyConfig<C, D>,
    pub stark: Starky<Chip<L>>,
    pub proof_target: StarkProofTarget<D>,
    pub public_input_targets: Vec<Target>,
    pub trace_generator: ArithmeticGenerator<L>,
}

impl<L: AirParameters, C, const D: usize> SimpleStarkWitnessGenerator<L, C, D> {
    pub fn new(
        config: StarkyConfig<C, D>,
        stark: Starky<Chip<L>>,
        proof_target: StarkProofTarget<D>,
        public_input_targets: Vec<Target>,
        trace_generator: ArithmeticGenerator<L>,
    ) -> Self {
        Self {
            config,
            stark,
            proof_target,
            public_input_targets,
            trace_generator,
        }
    }

    pub fn id() -> String {
        format!(
            "SimpleStarkWitnessGenerator, air parameters: {}, D = {}",
            L::id(),
            D
        )
        .to_string()
    }
}

impl<L: AirParameters, C, const D: usize> SimpleGenerator<L::Field, D>
    for SimpleStarkWitnessGenerator<L, C, D>
where
    L::Field: RichField + Extendable<D>,
    Chip<L>: Plonky2Air<L::Field, D>,
    C: CurtaConfig<D, F = L::Field>,
{
    fn id(&self) -> String {
        Self::id()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.public_input_targets.clone()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<L::Field>,
        out_buffer: &mut GeneratedValues<L::Field>,
    ) {
        let public_inputs = witness.get_targets(&self.public_input_targets);

        let proof = StarkyProver::<L::Field, C, D>::prove(
            &self.config,
            &self.stark,
            &self.trace_generator,
            &public_inputs,
        )
        .unwrap();

        set_stark_proof_target(out_buffer, &self.proof_target, &proof);

        self.trace_generator.reset();
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<L::Field, D>,
    ) -> IoResult<()> {
        let data = bincode::serialize(&self).unwrap();
        dst.write_bytes(&data)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<L::Field, D>,
    ) -> IoResult<Self>
    where
        Self: Sized,
    {
        let bytes = src.read_bytes()?;
        let data = bincode::deserialize(&bytes).unwrap();
        Ok(data)
    }
}
