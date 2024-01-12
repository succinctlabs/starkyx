use std::collections::HashMap;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::{
    NUM_BIT_OPPS, OPCODE_AND, OPCODE_INDICES, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR,
    OPCODE_SHR_CARRY, OPCODE_XOR,
};
use crate::chip::uint::bytes::register::ByteRegister;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::trace::AirTrace;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplicityData {
    pub multiplicities: ArrayRegister<ElementRegister>,
    operations_multipcitiy_dict: HashMap<ByteOperation<u8>, (usize, usize)>,
    pub operations_dict: HashMap<usize, Vec<ByteOperation<u8>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteMultiplicityData {
    data: MultiplicityData,
    trace_operations: Vec<ByteOperation<ByteRegister>>,
    public_operations: Vec<ByteOperation<ByteRegister>>,
}

impl MultiplicityData {
    pub fn new(multiplicities: ArrayRegister<ElementRegister>) -> Self {
        let mut operations_multipcitiy_dict = HashMap::new();
        let mut operations_dict = HashMap::new();
        for (row_index, (a, b)) in (0..=u8::MAX).cartesian_product(0..=u8::MAX).enumerate() {
            let mut operations = Vec::with_capacity(NUM_BIT_OPPS + 1);
            for (op_index, opcode) in OPCODE_INDICES.into_iter().enumerate() {
                let operation = match opcode {
                    OPCODE_AND => ByteOperation::and(a, b),
                    OPCODE_XOR => ByteOperation::xor(a, b),
                    OPCODE_SHR => ByteOperation::shr(a, b),
                    OPCODE_SHR_CARRY => ByteOperation::shr_full(a, b),
                    OPCODE_ROT => ByteOperation::rot(a, b),
                    OPCODE_NOT => ByteOperation::not(a),
                    OPCODE_RANGE => ByteOperation::range(a),
                    _ => unreachable!("Invalid opcode: {}", opcode),
                };
                operations_multipcitiy_dict.insert(operation, (row_index, op_index));
                operations.push(operation);
            }
            operations_dict.insert(row_index, operations);
        }
        Self {
            multiplicities,
            operations_dict,
            operations_multipcitiy_dict,
        }
    }

    pub fn update<F: Field>(&self, operation: &ByteOperation<u8>, writer: &TraceWriter<F>) {
        let (row, col) = self.operations_multipcitiy_dict[operation];
        writer.fetch_and_modify(&self.multiplicities.get(col), |x| *x + F::ONE, row);
    }

    pub fn multiplicities(&self) -> ArrayRegister<ElementRegister> {
        self.multiplicities
    }
}

impl ByteMultiplicityData {
    pub fn new(
        data: MultiplicityData,
        trace_operations: Vec<ByteOperation<ByteRegister>>,
        public_operations: Vec<ByteOperation<ByteRegister>>,
    ) -> Self {
        Self {
            data,
            trace_operations,
            public_operations,
        }
    }

    pub fn get_multiplicities<F: PrimeField64>(&self, writer: &TraceWriter<F>) -> AirTrace<F> {
        let mut multiplicities_trace = AirTrace::new_with_value(NUM_BIT_OPPS + 1, 1 << 16, 0u32);

        // Count the multiplicities in the trace
        let num_rows = writer.height;
        for i in 0..num_rows {
            for op in self.trace_operations.iter() {
                let op_value = op.read_from_writer(writer, i);
                let (row_index, col_index) = self.data.operations_multipcitiy_dict[&op_value];
                assert!(col_index < NUM_BIT_OPPS + 1);
                assert!(row_index < 1 << 16);
                multiplicities_trace.row_mut(row_index)[col_index] += 1;
            }
        }

        // Count the multiplicities in public inputs
        let public_slice = writer.public.read().unwrap();
        for op in self.public_operations.iter() {
            let op_value = op.read_from_slice(&public_slice);
            let (row_index, col_index) = self.data.operations_multipcitiy_dict[&op_value];
            assert!(col_index < NUM_BIT_OPPS + 1);
            assert!(row_index < 1 << 16);
            multiplicities_trace.row_mut(row_index)[col_index] += 1;
        }

        AirTrace::from_rows(
            multiplicities_trace
                .values
                .into_par_iter()
                .map(F::from_canonical_u32)
                .collect(),
            NUM_BIT_OPPS + 1,
        )
    }

    pub fn multiplicities<F: PrimeField64>(&self) -> ArrayRegister<ElementRegister> {
        self.data.multiplicities
    }
}
