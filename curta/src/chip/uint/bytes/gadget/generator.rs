use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::PartitionWitness;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult};

use super::air::ByteGadgetParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::Chip;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct BytesLookupGenerator<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>
{
    operations: Vec<ByteOperation<Target>>,
    air: Chip<ByteGadgetParameters<F, E, D>>,
}

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> SimpleGenerator<F, D>
    for BytesLookupGenerator<F, E, D>
{
    fn id(&self) -> String {
        "byte operation lookup".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let mut dependencies = Vec::new();
        for op in self.operations.iter() {
            match op {
                ByteOperation::And(a, b, _) => {
                    dependencies.push(*a);
                    dependencies.push(*b);
                }
                ByteOperation::Xor(a, b, _) => {
                    dependencies.push(*a);
                    dependencies.push(*b);
                }
                ByteOperation::Not(a, _) => {
                    dependencies.push(*a);
                }
                ByteOperation::Shr(a, b, _) => {
                    dependencies.push(*a);
                    dependencies.push(*b);
                }
                ByteOperation::ShrCarry(a, _, _, _) => {
                    dependencies.push(*a);
                }
                ByteOperation::ShrConst(a, _, _) => {
                    dependencies.push(*a);
                }
                ByteOperation::Rot(a, b, _) => {
                    dependencies.push(*a);
                }
                ByteOperation::RotConst(a, _, _) => {
                    dependencies.push(*a);
                }
                ByteOperation::Range(a) => {
                    dependencies.push(*a);
                }
            }
        }
        dependencies
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {}

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        unimplemented!("ByteOperation::serialize")
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        unimplemented!("ByteOperation::deserialize")
    }
}
