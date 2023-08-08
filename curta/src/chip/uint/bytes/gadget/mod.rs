use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;

use self::air::ByteGadgetParameters;
use super::operations::value::ByteOperation;
use crate::chip::builder::AirBuilder;

pub mod air;
pub mod generator;

use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct BytesGadget<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> {
    operations: Vec<ByteOperation<Target>>,
    air_builder: AirBuilder<ByteGadgetParameters<F, E, D>>,
}

pub struct ByteTarget(Target);
