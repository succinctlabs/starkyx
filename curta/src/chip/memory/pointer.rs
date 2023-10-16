use crate::chip::{AirParameters, register::{cubic::EvalCubic, Register}, builder::AirBuilder};

use core::hash::Hash;

pub trait Pointer<L: AirParameters> : 'static + Hash + Copy + Send + Sync {
    type Digest: EvalCubic;
    type Value : Register;

    fn compress(&self, builder: &mut AirBuilder<L>, value: &Self::Value) -> Self::Digest;
}