use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult};

use super::air::ByteGadgetParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::Chip;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct BytesLookupGenerator<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>
{
    operations: Vec<ByteOperation<Target>>,
    trace_generator: ArithmeticGenerator<ByteGadgetParameters<F, E, D>>,
    air: Chip<ByteGadgetParameters<F, E, D>>,
}

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> SimpleGenerator<F, D>
    for BytesLookupGenerator<F, E, D>
{
    fn id(&self) -> String {
        "byte operation lookup".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.operations
            .iter()
            .flat_map(|op| op.input_targets())
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        // let operations = self.operations.iter().map(|op|
        //     match op {
        //         ByteOperation::And(a, b, _) => {
        //             let a = witness.get_target(*a).as_canonical_u64() as u8;
        //             let b = witness.get_target(*b).as_canonical_u64() as u8;
        //             ByteOperation::and(a, b)
        //         }
        //         ByteOperation::Xor(a, b, _) => {
        //             let a = witness.get_target(*a).as_canonical_u64() as u8;
        //             let b = witness.get_target(*b).as_canonical_u64() as u8;
        //             ByteOperation::xor(a, b)
        //         }
        //         ByteOperation::Not(a, _) => {
        //             let a = witness.get_target(*a).as_canonical_u64() as u8;
        //             ByteOperation::not(a)
        //         }
        //         ByteOperation::Shr(a, b, _) => {
        //             let a = witness.get_target(*a).as_canonical_u64() as u8;
        //             let b = witness.get_target(*b).as_canonical_u64() as u8;
        //             ByteOperation::shr(a, b)
        //         }
        //         ByteOperation::ShrCarry(a, b, _, _) => {
        //             let a = witness.get_target(*a).as_canonical_u64() as u8;
        //             let b_mod = b & 0x7;
        //             let (res_val, carry_val) = if b_mod != 0 {
        //                 let res_val = a >> b_mod;
        //                 let carry_val = (a << (8 - b_mod)) >> (8 - b_mod);
        //                 debug_assert_eq!(
        //                     a.rotate_right(b_mod as u32),
        //                     res_val + (carry_val << (8 - b_mod))
        //                 );
        //                 (res_val, carry_val)
        //             } else {
        //                 (a, 0u8)
        //             };
        //             ByteOperation::Rot(a, *b, res_val + (carry_val << (8 - b_mod)))
        //         }
        //         ByteOperation::ShrConst(a, _, _) => {
        //             dependencies.push(*a);
        //         }
        //         ByteOperation::Rot(a, b, _) => {
        //             dependencies.push(*a);
        //         }
        //         ByteOperation::RotConst(a, _, _) => {
        //             dependencies.push(*a);
        //         }
        //         ByteOperation::Range(a) => {
        //             dependencies.push(*a);
        //         }
        //     }
        // ).collect::<Vec<_>>();
    }

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
