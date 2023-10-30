use std::collections::HashMap;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::table::log_derivative::entry::LogEntry;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::{
    NUM_BIT_OPPS, OPCODE_AND, OPCODE_INDICES, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR,
    OPCODE_XOR,
};
use crate::math::prelude::*;
use crate::trace::AirTrace;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplicityData {
    pub multiplicities: ArrayRegister<ElementRegister>,
    operations_multipcitiy_dict: HashMap<ByteOperation<u8>, (usize, usize)>,
    values_multiplicity_dict: HashMap<u32, (usize, usize)>,
    pub operations_dict: HashMap<usize, Vec<ByteOperation<u8>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteMultiplicityData {
    data: MultiplicityData,
    trace_values: Vec<LogEntry<ElementRegister>>,
    public_values: Vec<LogEntry<ElementRegister>>,
}

impl MultiplicityData {
    pub fn new(multiplicities: ArrayRegister<ElementRegister>) -> Self {
        let mut operations_multipcitiy_dict = HashMap::new();
        let mut values_multiplicity_dict = HashMap::new();
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
                let lookup_value = operation.lookup_digest_value();
                values_multiplicity_dict.insert(lookup_value, (row_index, op_index));
                operations.push(operation);
            }
            operations_dict.insert(row_index, operations);
        }
        Self {
            multiplicities,
            operations_dict,
            operations_multipcitiy_dict,
            values_multiplicity_dict,
        }
    }

    pub fn update<F: Field>(&self, operation: &ByteOperation<u8>, writer: &TraceWriter<F>) {
        let (row, col) = self.operations_multipcitiy_dict[operation];
        writer.fetch_and_modify(&self.multiplicities.get(col), |x| *x + F::ONE, row);
    }

    pub fn multiplicities(&self) -> ArrayRegister<ElementRegister> {
        self.multiplicities
    }

    pub fn table_index<F: PrimeField64>(&self, lookup_element: F) -> (usize, usize) {
        self.values_multiplicity_dict[&(lookup_element.as_canonical_u64() as u32)]
    }
}

impl ByteMultiplicityData {
    pub fn new(
        data: MultiplicityData,
        trace_values: Vec<LogEntry<ElementRegister>>,
        public_values: Vec<LogEntry<ElementRegister>>,
    ) -> Self {
        Self {
            data,
            trace_values,
            public_values,
        }
    }

    pub fn get_multiplicities<F: PrimeField64>(&self, writer: &TraceWriter<F>) -> AirTrace<F> {
        writer.get_multiplicities_from_fn(
            NUM_BIT_OPPS + 1,
            1 << 16,
            &self.trace_values,
            &self.public_values,
            |x| self.data.table_index(x),
        )
    }

    pub fn multiplicities<F: PrimeField64>(&self) -> ArrayRegister<ElementRegister> {
        self.data.multiplicities
    }
}
