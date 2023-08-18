use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult};

use super::air::ByteGadgetParameters;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::table::ByteLookupTable;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::math::prelude::*;

// A generator for the byte lookup STARK
#[derive(Debug, Clone)]
pub struct BytesLookupGenerator<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>
{
    operations: Vec<ByteOperation<Target>>,
    air_operations: Vec<ByteOperation<ByteRegister>>,
    trace_generator: ArithmeticGenerator<ByteGadgetParameters<F, E, D>>,
    table: ByteLookupTable,
}

pub struct ByteOperationGenerator {}

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>
    BytesLookupGenerator<F, E, D>
{
    /// Create a new instance from all the entries
    pub fn new(
        operations: Vec<ByteOperation<Target>>,
        air_operations: Vec<ByteOperation<ByteRegister>>,
        trace_generator: ArithmeticGenerator<ByteGadgetParameters<F, E, D>>,
        table: ByteLookupTable,
    ) -> Self {
        Self {
            operations,
            air_operations,
            trace_generator,
            table,
        }
    }

    pub fn hint(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
        writer: &TraceWriter<F>,
    ) {
        // Set all the target values and write the operation values to the trace
        for (op, air_op) in self.operations.iter().zip(self.air_operations.iter()) {
            match op {
                ByteOperation::And(a_t, b_t, res_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let b = witness.get_target(*b_t).as_canonical_u64() as u8;
                    let ByteOperation::And(a, b, res) = ByteOperation::and(a, b).as_field_op::<F>()
                    else {
                        unreachable!()
                    };
                    out_buffer.set_target(*res_t, res);

                    let ByteOperation::And(a_r, b_r, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    writer.write(a_r, &a, 0);
                    writer.write(b_r, &b, 0);
                }
                ByteOperation::Xor(a_t, b_t, res_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let b = witness.get_target(*b_t).as_canonical_u64() as u8;
                    let ByteOperation::Xor(a, b, res) = ByteOperation::xor(a, b).as_field_op::<F>()
                    else {
                        unreachable!()
                    };
                    out_buffer.set_target(*res_t, res);

                    let ByteOperation::Xor(a_r, b_r, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    writer.write(a_r, &a, 0);
                    writer.write(b_r, &b, 0);
                }
                ByteOperation::Not(a_t, res_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let ByteOperation::Not(a, res) = ByteOperation::not(a).as_field_op::<F>()
                    else {
                        unreachable!()
                    };
                    let ByteOperation::Not(a_r, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    out_buffer.set_target(*res_t, res);
                    writer.write(a_r, &a, 0);
                }
                ByteOperation::Shr(a_t, b_t, res_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let b = witness.get_target(*b_t).as_canonical_u64() as u8;
                    let ByteOperation::Shr(a, b, res) = ByteOperation::shr(a, b).as_field_op::<F>()
                    else {
                        unreachable!()
                    };
                    out_buffer.set_target(*res_t, res);

                    let ByteOperation::Shr(a_r, b_r, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    writer.write(a_r, &a, 0);
                    writer.write(b_r, &b, 0);
                }
                ByteOperation::ShrConst(a_t, b, res_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let res = F::from_canonical_u8(a >> (b & 0x7));
                    out_buffer.set_target(*res_t, res);

                    let ByteOperation::ShrConst(a_r, _, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    writer.write(a_r, &F::from_canonical_u8(a), 0);
                }
                ByteOperation::ShrCarry(a_t, b, res_t, c_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let b_mod = b & 0x7;
                    let (res, carry) = if b_mod != 0 {
                        (a >> b_mod, (a << (8 - b_mod)) >> (8 - b_mod))
                    } else {
                        (a, 0u8)
                    };
                    out_buffer.set_target(*res_t, F::from_canonical_u8(res));
                    out_buffer.set_target(*c_t, F::from_canonical_u8(carry));

                    let ByteOperation::ShrCarry(a_r, _, _, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    writer.write(a_r, &F::from_canonical_u8(a), 0);
                }
                ByteOperation::Rot(a_t, b_t, res_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let b = witness.get_target(*b_t).as_canonical_u64() as u8;
                    let ByteOperation::Rot(a, b, res) = ByteOperation::rot(a, b).as_field_op::<F>()
                    else {
                        unreachable!()
                    };
                    out_buffer.set_target(*res_t, res);

                    let ByteOperation::Rot(a_r, b_r, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    writer.write(a_r, &a, 0);
                    writer.write(b_r, &b, 0);
                }
                ByteOperation::RotConst(a_t, b, res_t) => {
                    let a = witness.get_target(*a_t).as_canonical_u64() as u8;
                    let ByteOperation::Rot(a, _, res) =
                        ByteOperation::rot(a, *b).as_field_op::<F>()
                    else {
                        unreachable!()
                    };
                    out_buffer.set_target(*res_t, res);

                    let ByteOperation::RotConst(a_r, _, _) = air_op else {
                        unreachable!("Air operations must match")
                    };
                    writer.write(a_r, &a, 0);
                }
                ByteOperation::Range(a_t) => {
                    let a = witness.get_target(*a_t);
                    let ByteOperation::Range(a_r) = air_op else {
                        unreachable!()
                    };
                    writer.write(a_r, &a, 0);
                }
            }
        }
    }
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
        let writer = self.trace_generator.new_writer();
        // Write the operation values
        self.hint(witness, out_buffer, &writer);

        // Write the operations and table multiplicities
        self.table.write_table_entries(&writer);
        for i in 0..(1 << 16) {
            writer.write_row_instructions(&self.trace_generator.air_data, i);
        }
        writer.write_global_instructions(&self.trace_generator.air_data);
        self.table.write_multiplicities(&writer);
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        unimplemented!("ByteOperation::serialize")
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        unimplemented!("ByteOperation::deserialize")
    }
}
