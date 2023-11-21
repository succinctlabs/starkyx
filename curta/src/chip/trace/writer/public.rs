use core::hash::Hash;

use super::AirWriter;
use crate::chip::memory::map::MemoryMap;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;

pub struct PublicWriter<'a, F: PartialEq + Eq + Hash> {
    public_values: &'a mut [F],
    memory: &'a mut MemoryMap<F>,
    height: usize,
}

impl<'a, F: PartialEq + Eq + Hash> PublicWriter<'a, F> {
    pub fn new(public_values: &'a mut [F], memory: &'a mut MemoryMap<F>, height: usize) -> Self {
        Self {
            public_values,
            memory,
            height,
        }
    }
}

impl<'a, F: Field> AirWriter for PublicWriter<'a, F> {
    type Field = F;

    fn read_slice(&self, memory_slice: &MemorySlice) -> &[F] {
        match memory_slice {
            MemorySlice::Public(index, length) => &self.public_values[*index..*index + *length],
            _ => panic!("Invalid memory slice for reading from public writer"),
        }
    }

    fn write_slice(&mut self, memory_slice: &MemorySlice, value: &[F]) {
        match memory_slice {
            MemorySlice::Public(index, length) => {
                self.public_values[*index..*index + *length].copy_from_slice(value);
            }
            _ => panic!("Can only write to public memory with public writer"),
        }
    }

    fn memory(&self) -> &MemoryMap<F> {
        self.memory
    }

    fn memory_mut(&mut self) -> &mut MemoryMap<F> {
        self.memory
    }

    fn row_index(&self) -> Option<usize> {
        None
    }

    fn height(&self) -> usize {
        self.height
    }
}
