use alloc::sync::Arc;
use core::borrow::Borrow;
use core::ops::Deref;
use std::sync::{LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::trace::window::TraceWindow;
use crate::trace::window_parser::TraceWindowParser;
use crate::trace::AirTrace;

#[derive(Debug)]
pub struct WriterData<T> {
    trace: RwLock<AirTrace<T>>,
    pub(crate) global: RwLock<Vec<T>>,
    pub(crate) public: RwLock<Vec<T>>,
    pub(crate) challenges: RwLock<Vec<T>>,
    height: usize,
}

#[derive(Debug, Clone)]
pub struct TraceWriter<T>(pub Arc<WriterData<T>>);

impl<T> TraceWriter<T> {
    #[inline]
    pub fn new(width: usize, num_rows: usize) -> Self {
        let height = num_rows;
        Self(Arc::new(WriterData {
            trace: RwLock::new(AirTrace::new_with_capacity(width, num_rows)),
            global: RwLock::new(Vec::new()),
            challenges: RwLock::new(Vec::new()),
            public: RwLock::new(Vec::new()),
            height,
        }))
    }

    #[inline]
    pub fn new_with_value(
        value: T,
        width: usize,
        num_rows: usize,
        num_public_inputs: usize,
        num_global_values: usize,
    ) -> Self
    where
        T: Copy,
    {
        let height = num_rows;
        Self(Arc::new(WriterData {
            trace: RwLock::new(AirTrace::new_with_value(width, num_rows, value)),
            global: RwLock::new(vec![value; num_global_values]),
            public: RwLock::new(vec![value; num_public_inputs]),
            challenges: RwLock::new(Vec::new()),
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

    pub fn global_mut(&self) -> LockResult<RwLockWriteGuard<'_, Vec<T>>> {
        self.0.global.write()
    }

    pub fn public_mut(&self) -> LockResult<RwLockWriteGuard<'_, Vec<T>>> {
        self.0.public.write()
    }
}

impl<F: Field> TraceWriter<F> {
    #[inline]
    pub fn read<R: Register>(&self, register: &R, row_index: usize) -> R::Value<F> {
        match register.register() {
            MemorySlice::Local(_, _) => self.read_from_trace(register, row_index),
            MemorySlice::Next(_, _) => self.read_from_trace(register, row_index),
            MemorySlice::Global(_, _) => self.read_from_global(register, row_index),
            MemorySlice::Public(_, _) => self.read_from_public(register, row_index),
            MemorySlice::Challenge(_, _) => self.read_from_challenge(register, row_index),
        }
    }

    #[inline]
    fn read_from_trace<R: Register>(&self, register: &R, row_index: usize) -> R::Value<F> {
        let trace = self.0.trace.read().unwrap();
        let window = trace.window(row_index);
        let parser = TraceWindowParser::new(window, &[], &[], &[]);
        register.eval(&parser)
    }

    #[inline]
    fn read_from_global<R: Register>(&self, register: &R, _row_index: usize) -> R::Value<F> {
        let global_values = self.0.global.read().unwrap();
        let window = TraceWindow::empty();
        let parser = TraceWindowParser::new(window, &[], &global_values, &[]);
        register.eval(&parser)
    }

    #[inline]
    fn read_from_challenge<R: Register>(&self, register: &R, _row_index: usize) -> R::Value<F> {
        let challenges = self.0.challenges.read().unwrap();
        let window = TraceWindow::empty();
        let parser = TraceWindowParser::new(window, &challenges, &[], &[]);
        register.eval(&parser)
    }

    #[inline]
    fn read_from_public<R: Register>(&self, register: &R, _row_index: usize) -> R::Value<F> {
        let public_inputs = self.0.public.read().unwrap();
        let window = TraceWindow::empty();
        let parser = TraceWindowParser::new(window, &[], &[], &public_inputs);
        register.eval(&parser)
    }

    #[inline]
    pub fn read_vec<R: Register>(
        &self,
        array: &ArrayRegister<R>,
        row_index: usize,
    ) -> Vec<R::Value<F>> {
        array
            .into_iter()
            .map(|register| self.read(&register, row_index))
            .collect()
    }

    #[inline]
    pub fn read_array<R: Register, const N: usize>(
        &self,
        array: &ArrayRegister<R>,
        row_index: usize,
    ) -> [R::Value<F>; N] {
        let elem_fn = |i| self.read(&array.get(i), row_index);
        core::array::from_fn(elem_fn)
    }

    #[inline]
    pub fn read_expression(
        &self,
        expression: &ArithmeticExpression<F>,
        row_index: usize,
    ) -> Vec<F> {
        let trace = self.0.trace.read().unwrap();
        let window = trace.window(row_index);
        let global_inputs = self.global.read().unwrap();
        let challenges = self.0.challenges.read().unwrap();
        let public_inputs = self.0.public.read().unwrap();
        let mut parser =
            TraceWindowParser::new(window, &challenges, &global_inputs, &public_inputs);
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
    pub fn write_slice<T: RegisterSerializable>(&self, data: &T, value: &[F], row_index: usize) {
        let mut trace = self.0.trace.write().unwrap();
        data.register()
            .assign(&mut trace.view_mut(), 0, value, row_index);
    }

    #[inline]
    pub fn write_array<T: Register, I>(&self, array: &ArrayRegister<T>, values: I, row_index: usize)
    where
        I: IntoIterator,
        I::Item: Borrow<T::Value<F>>,
    {
        for (data, value) in array.into_iter().zip(values) {
            self.write(&data, value.borrow(), row_index);
        }
    }

    #[inline]
    pub fn write<T: Register>(&self, data: &T, value: &T::Value<F>, row_index: usize) {
        match data.register() {
            MemorySlice::Local(..) => self.write_slice(data, T::align(value), row_index),
            MemorySlice::Next(..) => self.write_slice(data, T::align(value), row_index),
            MemorySlice::Global(..) => self.write_global(data, value),
            MemorySlice::Public(..) => {}
            MemorySlice::Challenge(..) => unreachable!("Challenge registers are read-only"),
        }
    }

    #[inline]
    fn write_global<T: Register>(&self, data: &T, value: &T::Value<F>) {
        match data.register() {
            MemorySlice::Global(_, _) => {
                let mut global = self.0.global.write().unwrap();
                data.assign_to_raw_slice(&mut global, value);
            }
            _ => panic!("Expected global register"),
        }
    }

    #[inline]
    pub fn write_instruction(&self, instruction: &impl Instruction<F>, row_index: usize) {
        instruction.write(self, row_index)
    }

    #[inline]
    pub fn write_row_instructions<L: AirParameters<Field = F>>(
        &self,
        air: &Chip<L>,
        row_index: usize,
    ) {
        for instruction in air.instructions.iter() {
            self.write_instruction(instruction, row_index);
        }
    }
}

impl<T> Deref for TraceWriter<T> {
    type Target = WriterData<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
