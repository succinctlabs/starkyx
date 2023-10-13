use super::AirWriter;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;

pub struct RowWriter<'a, F> {
    row: &'a mut [F],
    public_values: &'a [F],
}

impl<'a, F: Field> AirWriter<F> for RowWriter<'a, F> {
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
}
