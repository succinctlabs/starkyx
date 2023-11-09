use core::hash::Hash;

use super::AirWriter;
use crate::chip::memory::map::MemoryMap;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;

pub struct RowWriter<'a, F: PartialEq + Eq + Hash> {
    row: &'a mut [F],
    public_values: &'a [F],
    memory: &'a mut MemoryMap<F>,
    row_index: usize,
    height: usize,
}

impl<'a, F: PartialEq + Eq + Hash> RowWriter<'a, F> {
    pub fn new(
        row: &'a mut [F],
        public_values: &'a [F],
        memory: &'a mut MemoryMap<F>,
        row_index: usize,
        height: usize,
    ) -> Self {
        Self {
            row,
            public_values,
            memory,
            row_index,
            height,
        }
    }
}

impl<'a, F: Field> AirWriter for RowWriter<'a, F> {
    type Field = F;

    fn read_slice(&self, memory_slice: &MemorySlice) -> &[F] {
        match memory_slice {
            MemorySlice::Local(index, length) => &self.row[*index..*index + *length],
            MemorySlice::Public(index, length) => &self.public_values[*index..*index + *length],
            _ => panic!("Invalid memory slice for reading from row writer"),
        }
    }

    fn write_slice(&mut self, memory_slice: &MemorySlice, value: &[F]) {
        match memory_slice {
            MemorySlice::Local(index, length) => {
                self.row[*index..*index + *length].copy_from_slice(value);
            }
            _ => panic!("Invalid memory slice for writing with row writer"),
        }
    }

    fn memory(&self) -> &MemoryMap<F> {
        self.memory
    }

    fn memory_mut(&mut self) -> &mut MemoryMap<F> {
        self.memory
    }

    fn row_index(&self) -> Option<usize> {
        Some(self.row_index)
    }

    fn height(&self) -> usize {
        self.height
    }
}
