use core::hash::Hash;

use super::AirWriter;
use crate::chip::memory::map::MemoryMap;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;
use crate::trace::window::TraceWindowMut;

pub struct WindowWriter<'a, F: PartialEq + Eq + Hash> {
    window: TraceWindowMut<'a, F>,
    public_values: &'a [F],
    memory: &'a mut MemoryMap<F>,
}

impl<'a, F: Field> WindowWriter<'a, F> {
    pub fn new(
        window: TraceWindowMut<'a, F>,
        public_values: &'a [F],
        memory: &'a mut MemoryMap<F>,
    ) -> Self {
        Self {
            window,
            public_values,
            memory,
        }
    }
}

impl<'a, F: Field> AirWriter<F> for WindowWriter<'a, F> {
    fn read_slice(&self, memory_slice: &MemorySlice) -> &[F] {
        match memory_slice {
            MemorySlice::Local(index, length) => &self.window.local_slice[*index..*index + *length],
            MemorySlice::Next(index, length) => &self.window.next_slice[*index..*index + *length],
            MemorySlice::Public(index, length) => &self.public_values[*index..*index + *length],
            _ => panic!("Cannot read from challenges in window writer"),
        }
    }

    fn write_slice(&mut self, memory_slice: &MemorySlice, value: &[F]) {
        match memory_slice {
            MemorySlice::Local(index, length) => {
                self.window.local_slice[*index..*index + *length].copy_from_slice(value);
            }
            MemorySlice::Next(index, length) => {
                self.window.next_slice[*index..*index + *length].copy_from_slice(value);
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
}
