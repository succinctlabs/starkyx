use alloc::sync::Arc;
use core::ops::Deref;
use std::sync::{LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::air::parser::TraceWindowParser;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::*;
use crate::trace::AirTrace;

#[derive(Debug)]
pub struct WriterData<T> {
    trace: RwLock<AirTrace<T>>,
    public_inputs: Vec<T>,
    height: usize,
}

#[derive(Debug, Clone)]
pub struct TraceWriter<T>(pub Arc<WriterData<T>>);

impl<T> TraceWriter<T> {
    #[inline]
    pub fn new(width: usize, num_rows: usize, public_inputs: Vec<T>) -> Self {
        let height = num_rows;
        Self(Arc::new(WriterData {
            trace: RwLock::new(AirTrace::new_with_capacity(width, num_rows)),
            public_inputs,
            height,
        }))
    }

    #[inline]
    pub fn new_with_value(width: usize, num_rows: usize, value: T, public_inputs: Vec<T>) -> Self
    where
        T: Copy,
    {
        let height = num_rows;
        Self(Arc::new(WriterData {
            trace: RwLock::new(AirTrace::new_with_value(width, num_rows, value)),
            public_inputs,
            height,
        }))
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn read_trace(&self) -> LockResult<RwLockReadGuard<'_, AirTrace<T>>>
    where
        T: Clone,
    {
        self.0.trace.read()
    }

    pub fn write_trace(&self) -> LockResult<RwLockWriteGuard<'_, AirTrace<T>>> {
        self.0.trace.write()
    }
}

impl<F: Field> TraceWriter<F> {
    #[inline]
    pub fn read<R: Register>(&self, register: &R, row_index: usize) -> R::Value<F> {
        let trace = self.0.trace.read().unwrap();
        let window = trace.window(row_index);
        let parser = TraceWindowParser::new(window, &[], &self.public_inputs);
        register.eval(&parser)
    }

    pub fn read_expression(
        &self,
        expression: &ArithmeticExpression<F>,
        row_index: usize,
    ) -> Vec<F> {
        let trace = self.0.trace.read().unwrap();
        let window = trace.window(row_index);
        let mut parser = TraceWindowParser::new(window, &[], &self.public_inputs);
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

    #[inline]
    pub fn write_value<T: Register>(&self, data: &T, value: &T::Value<F>, row_index: usize) {
        self.write(data, T::align(value), row_index)
    }

    #[inline]
    pub fn write_instruction(&self, instruction: &impl Instruction<F>, row_index: usize) {
        instruction.write(self, row_index)
    }
}

impl<T> Deref for TraceWriter<T> {
    type Target = WriterData<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
