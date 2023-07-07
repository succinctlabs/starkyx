use alloc::sync::Arc;
use core::ops::Deref;
use std::sync::RwLock;

use super::StarkTrace;
use crate::air::parser::TraceWindowParser;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;

#[derive(Debug)]
pub struct WriterData<T> {
    trace: RwLock<StarkTrace<T>>,
    public_inputs: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct TraceWriter<T>(pub Arc<WriterData<T>>);

impl<T> TraceWriter<T> {
    #[inline]
    pub fn new(trace: StarkTrace<T>, public_inputs: Vec<T>) -> Self {
        Self(Arc::new(WriterData {
            trace: RwLock::new(trace),
            public_inputs,
        }))
    }
}

impl<F: Field> TraceWriter<F> {
    #[inline]
    pub fn read<R: Register>(&self, register: R, row_index: usize) -> R::Value<F> {
        let trace = self.0.trace.read().unwrap();
        let window = trace.window(row_index);
        let parser = TraceWindowParser::new(window, &self.0.public_inputs, &[]);
        register.eval(&parser)
    }

    #[inline]
    pub fn write_unsafe_batch_raw(
        &self,
        memory_slices: &[MemorySlice],
        values: &[F],
        row_index: usize,
    ) {
        let mut trace = self.0.trace.write().unwrap();
        memory_slices.iter().fold(0, |local_index, memory_slice| {
            memory_slice.assign(&mut trace.view_mut(), local_index, values, row_index)
        });
    }

    #[inline]
    pub fn write_unsafe_raw(&self, memory_slice: MemorySlice, value: &[F], row_index: usize) {
        let mut trace = self.0.trace.write().unwrap();
        memory_slice.assign(&mut trace.view_mut(), 0, value, row_index);
    }

    #[inline]
    pub fn write_batch<T: RegisterSerializable>(
        &self,
        data_slice: &[T],
        values: &[F],
        row_index: usize,
    ) {
        let mut trace = self.0.trace.write().unwrap();
        data_slice.iter().fold(0, |local_index, data| {
            data.register()
                .assign(&mut trace.view_mut(), local_index, values, row_index)
        });
    }

    #[inline]
    pub fn write<T: RegisterSerializable>(&self, data: &T, value: &[F], row_index: usize) {
        let mut trace = self.0.trace.write().unwrap();
        data.register()
            .assign(&mut trace.view_mut(), 0, value, row_index);
    }
}

impl<T> Deref for TraceWriter<T> {
    type Target = WriterData<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}