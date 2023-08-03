use std::collections::HashMap;
use std::sync::mpsc::Receiver;

use plonky2_maybe_rayon::ParallelIterator;

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::u32::opcode::{OPCODE_AND, OPCODE_XOR};
use crate::chip::uint::bytes::operations::instruction::ByteOperationValue;
use crate::chip::uint::bytes::operations::{NUM_BIT_OPPS, OPCODE_NOT, OPCODE_VALUES};
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
    pub fn update(&mut self, row: usize, col: usize) {
        self.0[row][col] += F::ONE;
    }
}

impl<F: Field> MultiplicityData<F> {
    pub fn new(
        rx: Receiver<ByteOperationValue<F>>,
        multiplicities: ArrayRegister<ElementRegister>,
    ) -> Self {
        for (i, (a, b)) in (0..=u8::MAX).zip(0..=u8::MAX).enumerate() {
            for op_index in 0..NUM_BIT_OPPS {
                let opcode = OPCODE_VALUES[op_index];

                for bit_input in [0, 1].iter() {
                    let operation = match (opcode, bit_input) {
                        (OPCODE_AND, _) => ByteOperationValue::and(a, b),
                        (OPCODE_XOR, _) => ByteOperationValue::xor(a, b),
                        (OPCODE_NOT, _) => ByteOperationValue::not(a),
                        (OPCODE_SHR, _) => ByteOperationValue::shr(a, *bit_input),
                        _ => unreachable!("Invalid opcode: {}", opcode),
                    };
                }
            }
        }

        todo!()
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
