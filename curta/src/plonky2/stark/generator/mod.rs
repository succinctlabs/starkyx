use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub mod simple;

use core::fmt::Debug;

use super::config::{CurtaConfig, StarkyConfig};
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::{AirParameters, Chip};

pub trait CurtaGenerator<L: AirParameters, C: CurtaConfig<D, F = L::Field>, const D: usize>:
    'static + Debug + Clone + Serialize + DeserializeOwned + Send + Sync
where
    L::Field: RichField + Extendable<D>,
{
    /// return the air that is being proven recursively by the generator.
    fn air(&self) -> &Chip<L>;

    /// writing the trace, public, and global values to the witness.
    fn write(&self, trace_generator: &ArithmeticGenerator<L>);

    // the config that is used for the proof.
    fn config(&self) -> StarkyConfig<C, D> {
        StarkyConfig::<C, D>::standard_fast_config(L::num_rows())
    }

    fn num_public_inputs(&self) -> usize {
        self.air().num_public_values
    }
}
