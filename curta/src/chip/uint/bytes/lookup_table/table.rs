use alloc::sync::Arc;
use core::array::from_fn;

use itertools::Itertools;

use super::super::operations::NUM_BIT_OPPS;
use super::multiplicity_data::MultiplicityData;
use super::ByteInstructionSet;
use crate::chip::bool::SelectInstruction;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::bit_operations::and::And;
use crate::chip::uint::bytes::bit_operations::not::Not;
use crate::chip::uint::bytes::bit_operations::util::u8_to_bits_le;
use crate::chip::uint::bytes::bit_operations::xor::Xor;
use crate::chip::uint::bytes::decode::ByteDecodeInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::{
    OPCODE_AND, OPCODE_INDICES, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR, OPCODE_XOR,
};
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::maybe_rayon::*;

#[derive(Debug, Clone)]
pub struct ByteLookupTable<F> {
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub results: [ByteRegister; NUM_BIT_OPPS],
    a_bits: ArrayRegister<BitRegister>,
    b_bits: ArrayRegister<BitRegister>,
    results_bits: [ArrayRegister<BitRegister>; NUM_BIT_OPPS],
    pub multiplicity_data: Arc<MultiplicityData>,
    pub digests: Vec<CubicRegister>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_byte_lookup_table(
        &mut self,
        row_acc_challenges: ArrayRegister<CubicRegister>,
    ) -> ByteLookupTable<L::Field>
    where
        L::Instruction: From<ByteInstructionSet>
            + From<SelectInstruction<BitRegister>>
            + From<ByteDecodeInstruction>,
    {
        let multiplicities = self.alloc_array::<ElementRegister>(NUM_BIT_OPPS + 1);

        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();
        let results = from_fn::<_, NUM_BIT_OPPS, _>(|_| self.alloc::<ByteRegister>());

        let a_bits = self.alloc_array::<BitRegister>(8);
        let b_bits = self.alloc_array::<BitRegister>(8);
        let results_bits = from_fn::<_, NUM_BIT_OPPS, _>(|_| self.alloc_array::<BitRegister>(8));

        let multiplicity_data = MultiplicityData::new(L::num_rows(), multiplicities);

        // Constrain the bit instructions
        for (k, &opcode) in OPCODE_INDICES.iter().enumerate() {
            match opcode {
                OPCODE_AND => {
                    let and = And {
                        a: a_bits,
                        b: b_bits,
                        result: results_bits[k],
                    };
                    self.register_instruction::<ByteInstructionSet>(and.into());
                }
                OPCODE_XOR => {
                    let xor = Xor {
                        a: a_bits,
                        b: b_bits,
                        result: results_bits[k],
                    };
                    self.register_instruction::<ByteInstructionSet>(xor.into());
                }
                OPCODE_NOT => {
                    let not = Not {
                        a: a_bits,
                        result: results_bits[k],
                    };
                    self.register_instruction::<ByteInstructionSet>(not.into());
                }
                OPCODE_SHR => {
                    self.set_shr(&a_bits, &b_bits.get_subarray(0..3), &results_bits[k]);
                }
                OPCODE_ROT => {
                    self.set_rotate_right(&a_bits, &b_bits.get_subarray(0..3), &results_bits[k]);
                }
                OPCODE_RANGE => {}
                _ => unreachable!("Invalid opcode"),
            }
        }

        // Constrain the equality between the byte registers and their bit representations
        self.decode_byte(&a, &a_bits);
        self.decode_byte(&b, &b_bits);
        for (result, bits) in results.iter().zip_eq(results_bits.iter()) {
            self.decode_byte(result, bits);
        }

        // Accumulate entries for the lookup table
        let mut digests = Vec::new();
        for (k, opcode) in OPCODE_INDICES.iter().enumerate() {
            let operation =
                ByteOperation::from_opcode_and_values(*opcode, a, b, results.get(k).copied());
            let acc_expressions = operation.expression_array();
            let digest = self.accumulate_expressions(&row_acc_challenges, &acc_expressions);
            digests.push(digest);
        }

        ByteLookupTable {
            a,
            b,
            results,
            a_bits,
            b_bits,
            results_bits,
            multiplicity_data: Arc::new(multiplicity_data),
            digests,
        }
    }
}

impl<F: PrimeField64> ByteLookupTable<F> {
    pub fn write_table_entries(&self, writer: &TraceWriter<F>) {
        let operations_dict = self.multiplicity_data.operations_dict.clone();
        // Write the lookup table entries
        writer
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .enumerate()
            .for_each(|(i, row)| {
                for (k, operation) in operations_dict[&i].iter().enumerate() {
                    let as_field_bits = |&x| u8_to_bits_le(x).map(|b| F::from_canonical_u8(b));
                    let as_field = |&x| F::from_canonical_u8(x);
                    match operation {
                        ByteOperation::And(a, b, c) => {
                            // Write field values
                            self.a.assign_to_raw_slice(row, &as_field(a));
                            self.b.assign_to_raw_slice(row, &as_field(b));
                            self.results[k].assign_to_raw_slice(row, &as_field(c));
                            // Write bit values
                            self.a_bits.assign_to_raw_slice(row, &as_field_bits(a));
                            self.b_bits.assign_to_raw_slice(row, &as_field_bits(b));
                            self.results_bits[k].assign_to_raw_slice(row, &as_field_bits(c));
                        }
                        ByteOperation::Xor(_, _, c) => {
                            // Write field values
                            self.results[k].assign_to_raw_slice(row, &as_field(c));
                            // Write bit values
                            self.results_bits[k].assign_to_raw_slice(row, &as_field_bits(c));
                        }
                        ByteOperation::Not(_, c) => {
                            // Write field values
                            self.results[k].assign_to_raw_slice(row, &as_field(c));
                            // Write bit values
                            self.results_bits[k].assign_to_raw_slice(row, &as_field_bits(c));
                        }
                        ByteOperation::Shr(_, _, c) => {
                            // Write field value
                            self.results[k].assign_to_raw_slice(row, &as_field(c));
                        }
                        ByteOperation::Rot(_, _, c) => {
                            // Write field value
                            self.results[k].assign_to_raw_slice(row, &as_field(c));
                        }

                        ByteOperation::Range(_) => {}
                        _ => unreachable!("const parameter operations are not supported"),
                    }
                }
            });
    }

    pub fn write_multiplicities(&self, writer: &TraceWriter<F>) {
        // Assign multiplicities to the trace
        self.multiplicity_data.write_multiplicities(writer);
    }
}
