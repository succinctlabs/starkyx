use core::array::from_fn;
use std::sync::mpsc::Receiver;

use super::super::operations::NUM_BIT_OPPS;
use super::multiplicity_data::MultiplicityData;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::OPCODE_INDICES;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::maybe_rayon::*;

#[derive(Debug)]
pub struct ByteLookupTable<F> {
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub results: [ByteRegister; NUM_BIT_OPPS],
    a_bits: ArrayRegister<BitRegister>,
    b_bits: ArrayRegister<BitRegister>,
    results_bits: [ArrayRegister<BitRegister>; NUM_BIT_OPPS],
    pub multiplicity_data: MultiplicityData<F>,
    pub digests: Vec<CubicRegister>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_byte_lookup_table(
        &mut self,
        row_acc_challenges: ArrayRegister<CubicRegister>,
        rx: Receiver<ByteOperation<u8>>,
    ) -> ByteLookupTable<L::Field> {
        let multiplicities = self.alloc_array::<ElementRegister>(NUM_BIT_OPPS);

        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();
        let results = from_fn::<_, NUM_BIT_OPPS, _>(|_| self.alloc::<ByteRegister>());

        let a_bits = self.alloc_array::<BitRegister>(8);
        let b_bits = self.alloc_array::<BitRegister>(8);
        let results_bits = from_fn::<_, NUM_BIT_OPPS, _>(|_| self.alloc_array::<BitRegister>(8));

        let multiplicity_data = MultiplicityData::new(L::num_rows(), rx, multiplicities);

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
            multiplicity_data,
            digests,
        }
    }
}

impl<F: PrimeField64> ByteLookupTable<F> {
    pub fn write(&mut self, writer: &TraceWriter<F>) {
        // Collect the multiplicity values
        self.multiplicity_data.collect_values();

        // Assign multiplicities to the trace
        self.multiplicity_data.write_multiplicities(writer);

        writer
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .enumerate()
            .for_each(|(i, row)| {
                for (k, operation) in self.multiplicity_data.operations_dict[&i]
                    .iter()
                    .enumerate()
                {
                    let field_op = operation.as_field_bits_op::<F>();
                    match field_op {
                        ByteOperation::And(a, b, c) => {
                            self.a_bits.assign_to_raw_slice(row, &a);
                            self.b_bits.assign_to_raw_slice(row, &b);
                            self.results_bits[k].assign_to_raw_slice(row, &c);
                        }
                        ByteOperation::Xor(a, b, c) => {
                            self.a_bits.assign_to_raw_slice(row, &a);
                            self.b_bits.assign_to_raw_slice(row, &b);
                            self.results_bits[k].assign_to_raw_slice(row, &c);
                        }
                        ByteOperation::Shr(a, b, c) => {
                            self.a_bits.assign_to_raw_slice(row, &a);
                            self.b_bits.assign_to_raw_slice(row, &b);
                            self.results_bits[k].assign_to_raw_slice(row, &c);
                        }
                        ByteOperation::Rot(a, b, c) => {
                            self.a_bits.assign_to_raw_slice(row, &a);
                            self.b_bits.assign_to_raw_slice(row, &b);
                            self.results_bits[k].assign_to_raw_slice(row, &c);
                        }
                        ByteOperation::Not(a, b) => {
                            self.a_bits.assign_to_raw_slice(row, &a);
                            self.results_bits[k].assign_to_raw_slice(row, &b);
                        }
                        ByteOperation::Range(a) => {
                            self.a_bits.assign_to_raw_slice(row, &a);
                        }
                    }
                }
            });
    }
}
