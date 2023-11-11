use core::hash::Hash;

use super::AirWriter;
use crate::chip::memory::map::MemoryMap;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;
use crate::trace::window::TraceWindowMut;

pub struct WindowWriter<'a, F: PartialEq + Eq + Hash> {
    pub(crate) window: TraceWindowMut<'a, F>,
    public_values: &'a [F],
    memory: &'a mut MemoryMap<F>,
    current_row: usize,
    height: usize,
}

impl<'a, F: PartialEq + Eq + Hash> WindowWriter<'a, F> {
    pub fn new(
        window: TraceWindowMut<'a, F>,
        public_values: &'a [F],
        memory: &'a mut MemoryMap<F>,
        current_row: usize,
        height: usize,
    ) -> Self {
        Self {
            window,
            public_values,
            memory,
            current_row,
            height,
        }
    }
}
impl<'a, F: Field> AirWriter for WindowWriter<'a, F> {
    type Field = F;

    fn read_slice(&self, memory_slice: &MemorySlice) -> &[F] {
        match memory_slice {
            MemorySlice::Local(index, length) => &self.window.local_slice[*index..*index + *length],
            MemorySlice::Public(index, length) => &self.public_values[*index..*index + *length],
            _ => panic!("Can only read from local and public registers using window writer"),
        }
    }

    fn write_slice(&mut self, memory_slice: &MemorySlice, value: &[F]) {
        match memory_slice {
            MemorySlice::Local(index, length) => {
                self.window.local_slice[*index..*index + *length].copy_from_slice(value);
            }
            MemorySlice::Next(index, length) => {
                if !self.window.is_last_row {
                    self.window.next_slice[*index..*index + *length].copy_from_slice(value);
                }
            }
            _ => panic!("Window writer cannot write to non trace values"),
        }
    }

    fn memory(&self) -> &MemoryMap<F> {
        self.memory
    }

    fn memory_mut(&mut self) -> &mut MemoryMap<F> {
        self.memory
    }

    fn row_index(&self) -> Option<usize> {
        Some(self.current_row)
    }

    fn height(&self) -> usize {
        self.height
    }
}
