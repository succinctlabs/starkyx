use log::{log, Level};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::IoResult;

use crate::math::field::PrimeField64;

pub trait Watcher {
    fn watch(&mut self, log: &str, targets: Vec<Target>);
}

impl<F: RichField + Extendable<D>, const D: usize> Watcher for CircuitBuilder<F, D> {
    fn watch(&mut self, log: &str, targets: Vec<Target>) {
        let log = String::from(log);

        let generator: WatchGenerator = WatchGenerator { targets, log };
        self.add_simple_generator(generator);
    }
}

#[derive(Debug, Clone)]
pub struct WatchGenerator {
    pub targets: Vec<Target>,
    pub log: String,
}

impl WatchGenerator {
    pub fn id() -> String {
        "WatchGenerator".to_string()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for WatchGenerator {
    fn id(&self) -> String {
        Self::id()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.targets.clone()
    }

    #[allow(unused_variables)]
    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn deserialize(
        src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Self>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, _out_buffer: &mut GeneratedValues<F>) {
        let values: Vec<u64> = self
            .targets
            .iter()
            .map(|x| witness.get_target(*x).as_canonical_u64())
            .collect();
        let formatted_log = if values.len() == 1 {
            format!("[Watch] {}: {:?}", self.log, values[0])
        } else {
            format!("[Watch] {}: {:?}", self.log, values)
        };
        log!(Level::Info, "{}", formatted_log);
    }
}
