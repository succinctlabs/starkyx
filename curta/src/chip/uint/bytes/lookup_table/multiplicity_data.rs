use std::collections::HashMap;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::{
    NUM_BIT_OPPS, OPCODE_AND, OPCODE_INDICES, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR,
    OPCODE_XOR,
};
use crate::math::prelude::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct MultiplicityData {
    multiplicities: ArrayRegister<ElementRegister>,
    operations_multipcitiy_dict: HashMap<ByteOperation<u8>, (usize, usize)>,
    pub operations_dict: HashMap<usize, Vec<ByteOperation<u8>>>,
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

    pub fn multiplicities(&self) -> &ArrayRegister<ElementRegister> {
        &self.multiplicities
    }
}
