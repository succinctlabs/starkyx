use serde::{Deserialize, Serialize};

use super::super::operations::NUM_BIT_OPPS;
use super::multiplicity_data::MultiplicityData;
use super::ByteInstructionSet;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::table::lookup::table::LogLookupTable;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::decode::ByteDecodeInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::{
    OPCODE_AND, OPCODE_INDICES, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR, OPCODE_SHR_CARRY,
    OPCODE_XOR,
};
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::maybe_rayon::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteLogLookupTable<F, E> {
    pub challenges: ArrayRegister<CubicRegister>,
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub a_and_b: ByteRegister,
    pub a_xor_b: ByteRegister,
    pub a_shr_b: ByteRegister,
    pub a_shr_carry_b: ByteRegister,
    pub a_rot_b: ByteRegister,
    pub a_not: ByteRegister,
    pub multiplicity_data: MultiplicityData,
    pub digests: Vec<CubicRegister>,
    pub lookup: LogLookupTable<CubicRegister, F, E>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_byte_lookup_table(&mut self) -> ByteLogLookupTable<L::Field, L::CubicParams>
    where
        L::Instruction: From<ByteInstructionSet> + From<ByteDecodeInstruction>,
    {
        let multiplicities = self.alloc_array::<ElementRegister>(NUM_BIT_OPPS + 1);

        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();

        let a_and_b = self.alloc::<ByteRegister>();
        let a_xor_b = self.alloc::<ByteRegister>();
        let a_shr_b = self.alloc::<ByteRegister>();
        let a_shr_carry_b = self.alloc::<ByteRegister>();
        let a_rot_b = self.alloc::<ByteRegister>();
        let a_not = self.alloc::<ByteRegister>();

        let multiplicity_data = MultiplicityData::new(multiplicities);

        // Accumulate entries for the lookup table
        let challenges = self.challenge_powers(5);

        let digests = OPCODE_INDICES
            .into_iter()
            .map(|op| {
                let operation = match op {
                    OPCODE_AND => ByteOperation::And(a, b, a_and_b),
                    OPCODE_XOR => ByteOperation::Xor(a, b, a_xor_b),
                    OPCODE_SHR => ByteOperation::Shr(a, b, a_shr_b),
                    OPCODE_SHR_CARRY => ByteOperation::ShrFull(a, b, a_shr_b, a_shr_carry_b),
                    OPCODE_ROT => ByteOperation::Rot(a, b, a_rot_b),
                    OPCODE_NOT => ByteOperation::Not(a, a_not),
                    OPCODE_RANGE => ByteOperation::Range(a),
                    _ => unreachable!("Invalid opcode: {}", op),
                };
                let values = operation.expressions();
                self.accumulate_expressions(&challenges, &values)
            })
            .collect::<Vec<_>>();

        let lookup = self.new_lookup(&digests, &multiplicities);

        ByteLogLookupTable {
            challenges,
            a,
            b,
            a_and_b,
            a_xor_b,
            a_shr_b,
            a_shr_carry_b,
            a_rot_b,
            a_not,
            multiplicity_data,
            digests,
            lookup,
        }
    }
}

impl<F: PrimeField64, E: CubicParameters<F>> ByteLogLookupTable<F, E> {
    pub fn multiplicities(&self) -> ArrayRegister<ElementRegister> {
        self.multiplicity_data.multiplicities
    }
    pub fn write_table_entries(&self, writer: &TraceWriter<F>) {
        let operations_dict = &self.multiplicity_data.operations_dict;
        // Write the lookup table entries
        writer
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .enumerate()
            .for_each(|(i, row)| {
                for operation in operations_dict[&i].iter() {
                    let as_field = |&x| F::from_canonical_u8(x);
                    match operation {
                        ByteOperation::And(a, b, c) => {
                            // Write field values
                            self.a.assign_to_raw_slice(row, &as_field(a));
                            self.b.assign_to_raw_slice(row, &as_field(b));
                            self.a_and_b.assign_to_raw_slice(row, &as_field(c));
                        }
                        ByteOperation::Xor(_, _, c) => {
                            // Write field values
                            self.a_xor_b.assign_to_raw_slice(row, &as_field(c));
                        }
                        ByteOperation::Not(_, c) => {
                            // Write field values
                            self.a_not.assign_to_raw_slice(row, &as_field(c));
                        }
                        ByteOperation::Shr(_, _, c) => {
                            // Write field value
                            self.a_shr_b.assign_to_raw_slice(row, &as_field(c));
                        }
                        ByteOperation::ShrFull(_, _, r, c) => {
                            // Write field value
                            self.a_shr_b.assign_to_raw_slice(row, &as_field(r));
                            self.a_shr_carry_b.assign_to_raw_slice(row, &as_field(c));
                        }
                        ByteOperation::Rot(_, _, c) => {
                            // Write field value
                            self.a_rot_b.assign_to_raw_slice(row, &as_field(c));
                        }
                        ByteOperation::Range(_) => {}
                        _ => unreachable!("const parameter operations are not supported"),
                    }
                }
            });
    }
}
