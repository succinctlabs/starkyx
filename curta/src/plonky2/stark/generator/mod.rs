use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::PartitionWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub mod simple;

use core::any::type_name;
use core::fmt::Debug;

use self::simple::SimpleStarkWitnessGenerator;
use super::config::{CurtaConfig, StarkyConfig};
use super::gadget::StarkGadget;
use super::Starky;
use crate::chip::builder::AirTraceData;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::{AirParameters, Chip};
use crate::plonky2::Plonky2Air;
use crate::utils::serde::{BufferRead, BufferWrite};

/// An interface for recursively verifying a Curta Starky AIR in plonky2.
pub trait CurtaGenerator<F: RichField + Extendable<D>, const D: usize>:
    'static + Debug + Clone + Serialize + DeserializeOwned + Send + Sync
{
    // The air parameters of the underlying chip.
    type AirParameters: AirParameters<Field = F>;

    // The curta config for the stark.
    type Config: CurtaConfig<D, F = F, FE = F::Extension>;

    /// return the air that is being proven recursively by the generator.
    fn build() -> (Chip<Self::AirParameters>, AirTraceData<Self::AirParameters>);

    /// the targets needed as an input for the air trace.
    fn inputs(&self) -> Vec<Target>;

    /// Targets for the public input slice of the stark.
    fn public_input_targets(&self) -> Vec<Target>;

    /// Writing the trace, public, and global values to the witness.
    fn write(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
        trace_generator: &ArithmeticGenerator<Self::AirParameters>,
    );

    /// A unique identifier for the generator type.
    ///
    /// By default, the `type_name` is used.
    fn id() -> String {
        format!("Curta Generator: {}", type_name::<Self>()).to_string()
    }

    // The config that is used for the proof.
    fn config(&self) -> StarkyConfig<Self::Config, D> {
        StarkyConfig::<Self::Config, D>::standard_fast_config(Self::AirParameters::num_rows())
    }

    /// Verifies the Curta Starky AIR recursively.
    fn verify_curta_stark(self, builder: &mut CircuitBuilder<F, D>)
    where
        Chip<Self::AirParameters>: Plonky2Air<F, D>,
    {
        let (air, trace_data) = Self::build();
        let trace_generator = ArithmeticGenerator::new(trace_data);

        let config = self.config();
        let stark = Starky::new(air);

        let public_input_targets = self.public_input_targets();

        let generator = CurtaSimpleGenerator::new(self, trace_generator.clone());
        builder.add_simple_generator(generator);

        let virtual_proof = builder.add_virtual_stark_proof(&stark, &config);
        builder.verify_stark_proof(
            &config,
            &stark,
            virtual_proof.clone(),
            &public_input_targets,
        );

        let stark_generator = SimpleStarkWitnessGenerator::new(
            config,
            stark,
            virtual_proof,
            public_input_targets,
            trace_generator,
        );

        builder.add_simple_generator(stark_generator);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CurtaSimpleGenerator<
    F: RichField + Extendable<D>,
    W: CurtaGenerator<F, D>,
    const D: usize,
> {
    pub inner: W,
    trace_generator: ArithmeticGenerator<W::AirParameters>,
}

impl<F: RichField + Extendable<D>, W: CurtaGenerator<F, D>, const D: usize>
    CurtaSimpleGenerator<F, W, D>
{
    pub fn new(inner: W, trace_generator: ArithmeticGenerator<W::AirParameters>) -> Self {
        Self {
            inner,
            trace_generator,
        }
    }
}

impl<F: RichField + Extendable<D>, W: CurtaGenerator<F, D>, const D: usize> SimpleGenerator<F, D>
    for CurtaSimpleGenerator<F, W, D>
{
    fn id(&self) -> String {
        "CurtaSimpleGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.inner.inputs()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        let data = bincode::serialize(&self).unwrap();
        dst.write_bytes(&data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        let bytes = src.read_bytes()?;
        let data = bincode::deserialize(&bytes).unwrap();
        Ok(data)
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        self.inner.write(witness, out_buffer, &self.trace_generator);
    }
}
