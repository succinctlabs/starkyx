use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;

use self::air::ByteGadgetParameters;
use super::operations::value::ByteOperation;
use crate::chip::builder::AirBuilder;

pub mod air;
pub mod generator;
pub mod operation;

use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct BytesGadget {
    operations: Vec<ByteOperation<Target>>,
}

pub struct ByteTarget(Target);
