use std::collections::HashMap;
use std::sync::mpsc::Receiver;

use plonky2_maybe_rayon::ParallelIterator;

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::operations::instruction::ByteOperationValue;
use crate::chip::uint::bytes::operations::{
    NUM_BIT_OPPS, OPCODE_AND, OPCODE_INDICES, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR,
    OPCODE_XOR,
};
use crate::math::prelude::*;
use crate::maybe_rayon::*;

#[derive(Debug, Clone)]
pub struct MultiplicityValues<T>(Vec<[T; NUM_BIT_OPPS]>);

#[derive(Debug)]
pub struct MultiplicityData<T> {
    rx: Receiver<ByteOperationValue<T>>,
    multiplicities: ArrayRegister<ElementRegister>,
    multiplicities_values: MultiplicityValues<T>,
    operations_dict: HashMap<ByteOperationValue<T>, (usize, usize)>,
}

impl<F: Field> MultiplicityValues<F> {
    pub fn new(num_rows: usize) -> Self {
        Self(vec![[F::ZERO; NUM_BIT_OPPS]; num_rows])
    }

    pub fn update(&mut self, row: usize, col: usize) {
        self.0[row][col] += F::ONE;
    }
}

impl<F: Field> MultiplicityData<F> {
    pub fn new(
        num_rows: usize,
        rx: Receiver<ByteOperationValue<F>>,
        multiplicities: ArrayRegister<ElementRegister>,
    ) -> Self {
        let mut operations_dict = HashMap::new();
        for (row_index, (a, b)) in (0..=u8::MAX).zip(0..=u8::MAX).enumerate() {
            for (op_index, opcode) in OPCODE_INDICES.into_iter().enumerate() {
                let operation = match opcode {
                    OPCODE_AND => ByteOperationValue::and(a, b).as_field_op(),
                    OPCODE_XOR => ByteOperationValue::xor(a, b).as_field_op(),
                    OPCODE_SHR => ByteOperationValue::shr(a, b).as_field_op(),
                    OPCODE_ROT => ByteOperationValue::rot(a, b).as_field_op(),
                    OPCODE_NOT => ByteOperationValue::not(a).as_field_op(),
                    OPCODE_RANGE => ByteOperationValue::range(a).as_field_op(),
                    _ => unreachable!("Invalid opcode: {}", opcode),
                };
                operations_dict.insert(operation, (row_index, op_index));
            }
        }
        let multiplicity_values = MultiplicityValues::new(num_rows);

        Self {
            rx,
            multiplicities,
            multiplicities_values: multiplicity_values,
            operations_dict,
        }
    }

    pub fn collect_values(&mut self) {
        for operation in self.rx.iter() {
            let (row, col) = self.operations_dict[&operation];
            self.multiplicities_values.update(row, col);
        }
    }

    pub fn assign(&self, writer: TraceWriter<F>) {
        let multiplicities_array = self.multiplicities;
        writer
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .zip_eq(self.multiplicities_values.0.par_iter())
            .for_each(|(row, multiplicities)| {
                multiplicities_array.assign_to_raw_slice(row, multiplicities);
            });
    }
}
