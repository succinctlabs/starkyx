use alloc::sync::Arc;
use core::ops::Deref;
use std::sync::RwLock;

use crate::air::parser::TraceWindowParser;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;
use crate::trace::StarkTrace;

#[derive(Debug)]
pub struct WriterData<T> {
    trace: RwLock<StarkTrace<T>>,
    challenges: Vec<T>,
    public_inputs: Vec<T>,
    height: usize,
}

#[derive(Debug, Clone)]
pub struct TraceWriter<T>(pub Arc<WriterData<T>>);

impl<T> TraceWriter<T> {
    #[inline]
    pub fn new(trace: StarkTrace<T>, public_inputs: Vec<T>, challenges: Vec<T>) -> Self {
        let height = trace.height();
        Self(Arc::new(WriterData {
            trace: RwLock::new(trace),
            public_inputs,
            challenges,
            height,
        }))
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }
}

impl<F: Field> TraceWriter<F> {
    #[inline]
    pub fn read<R: Register>(&self, register: R, row_index: usize) -> R::Value<F> {
        let trace = self.0.trace.read().unwrap();
        let window = trace.window(row_index);
        let parser = TraceWindowParser::new(window, &self.0.challenges, &self.0.public_inputs);
        register.eval(&parser)
    }

    pub fn read_expression(
        &self,
        expression: &ArithmeticExpression<F>,
        row_index: usize,
    ) -> Vec<F> {
        let trace = self.0.trace.read().unwrap();
        let window = trace.window(row_index);
        let mut parser = TraceWindowParser::new(window, &&self.0.challenges, &self.0.public_inputs);
        expression.eval(&mut parser)
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