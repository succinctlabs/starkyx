use super::AirWriter;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;
use crate::trace::window::TraceWindowMut;
use crate::trace::AirTrace;

pub struct WindowWriter<'a, F> {
    window: TraceWindowMut<'a, F>,
    public_values: &'a [F],
}

impl<'a, F: Field> WindowWriter<'a, F> {
    pub fn new(window: TraceWindowMut<'a, F>, public_values: &'a [F]) -> Self {
        Self {
            window,
            public_values,
        }
    }

    pub fn iter(
        trace: &'a mut AirTrace<F>,
        public_values: &'a [F],
    ) -> impl Iterator<Item = WindowWriter<'a, F>> + 'a {
        trace
            .windows_mut()
            .map(move |window| Self::new(window, public_values))
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
}
