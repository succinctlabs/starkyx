use alloc::sync::Arc;
use core::borrow::Borrow;
use core::hash::Hash;
use core::ops::Deref;
use std::sync::{LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use super::data::AirTraceData;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::instruction::Instruction;
use crate::chip::memory::map::MemoryMap;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::log_derivative::entry::{LogEntry, LogEntryValue};
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::trace::window::TraceWindow;
use crate::trace::window_parser::TraceWindowParser;
use crate::trace::AirTrace;

pub mod data;
pub mod public;
pub mod row;
pub mod window;

pub trait AirWriter: Sized {
    type Field: Field;

    fn write_slice(&mut self, memory_slice: &MemorySlice, value: &[Self::Field]);

    fn read_slice(&self, memory_slice: &MemorySlice) -> &[Self::Field];

    fn memory(&self) -> &MemoryMap<Self::Field>;

    fn memory_mut(&mut self) -> &mut MemoryMap<Self::Field>;

    /// The current row index in the trace of the writer.
    fn row_index(&self) -> Option<usize>;

    // The total number of lines in the trace.
    fn height(&self) -> usize;

    fn read<T: Register>(&self, data: &T) -> T::Value<Self::Field> {
        let slice = self.read_slice(data.register());
        T::value_from_slice(slice)
    }

    fn read_expression(&self, expression: &ArithmeticExpression<Self::Field>) -> Vec<Self::Field> {
        expression.expression.eval_writer(self)
    }

    fn read_vec<R: Register>(&self, array: &ArrayRegister<R>) -> Vec<R::Value<Self::Field>> {
        array
            .into_iter()
            .map(|register| self.read(&register))
            .collect()
    }

    fn read_array<R: Register, const N: usize>(
        &self,
        array: &ArrayRegister<R>,
    ) -> [R::Value<Self::Field>; N] {
        let elem_fn = |i| self.read(&array.get(i));
        core::array::from_fn(elem_fn)
    }

    fn write<T: Register>(&mut self, data: &T, value: &T::Value<Self::Field>) {
        self.write_slice(data.register(), T::align(value))
    }

    fn write_array<T: Register, I>(&mut self, array: &ArrayRegister<T>, values: I)
    where
        I: IntoIterator,
        I::Item: Borrow<T::Value<Self::Field>>,
    {
        for (data, value) in array.into_iter().zip(values) {
            self.write(&data, value.borrow());
        }
    }

    fn write_instruction(&mut self, instruction: &impl Instruction<Self::Field>) {
        instruction.write_to_air(self)
    }

    fn read_log_entry<T: EvalCubic>(&self, entry: &LogEntry<T>) -> LogEntryValue<Self::Field> {
        let eval = |value: &T| T::trace_value_as_cubic(self.read(value));
        match entry {
            LogEntry::Input(value) => LogEntryValue {
                value: eval(value),
                multiplier: Self::Field::ONE,
            },
            LogEntry::Output(value) => LogEntryValue {
                value: eval(value),
                multiplier: -Self::Field::ONE,
            },
            LogEntry::InputMultiplicity(value, multiplier) => {
                let multiplier = self.read(multiplier);
                LogEntryValue {
                    value: eval(value),
                    multiplier,
                }
            }
            LogEntry::OutputMultiplicity(value, multiplier) => {
                let multiplier = self.read(multiplier);
                LogEntryValue {
                    value: eval(value),
                    multiplier: -multiplier,
                }
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WriterData<T: PartialEq + Eq + Hash> {
    pub(crate) trace: RwLock<AirTrace<T>>,
    pub(crate) global: RwLock<Vec<T>>,
    pub(crate) public: RwLock<Vec<T>>,
    pub(crate) challenges: RwLock<Vec<T>>,
    pub(crate) memory: RwLock<MemoryMap<T>>,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerWriterData<F> {
    pub trace: AirTrace<F>,
    pub global: Vec<F>,
    pub public: Vec<F>,
    pub challenges: Vec<F>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceWriter<T: PartialEq + Eq + Hash>(pub Arc<WriterData<T>>);

impl<T: PartialEq + Eq + Hash> TraceWriter<T> {
    #[inline]
    pub fn new<L: AirParameters<Field = T>>(air_data: &AirTraceData<L>, num_rows: usize) -> Self
    where
        T: Field,
    {
        let num_public_inputs = air_data.num_public_inputs;
        let num_global_values = air_data.num_global_values;
        Self::new_with_value(
            L::Field::ZERO,
            L::num_columns(),
            num_rows,
            num_public_inputs,
            num_global_values,
        )
    }

    pub fn into_inner(self) -> Result<InnerWriterData<T>>
    where
        T: 'static + Send + Sync,
    {
        let WriterData {
            trace,
            global,
            public,
            challenges,
            ..
        } = Arc::into_inner(self.0).ok_or(anyhow!("Arc unpacking failed"))?;

        Ok(InnerWriterData {
            trace: RwLock::into_inner(trace)?,
            global: RwLock::into_inner(global)?,
            public: RwLock::into_inner(public)?,
            challenges: RwLock::into_inner(challenges)?,
        })
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
            memory: RwLock::new(MemoryMap::new()),
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

    pub fn global(&self) -> LockResult<RwLockReadGuard<'_, Vec<T>>> {
        self.0.global.read()
    }

    pub fn public_mut(&self) -> LockResult<RwLockWriteGuard<'_, Vec<T>>> {
        self.0.public.write()
    }

    pub fn public(&self) -> LockResult<RwLockReadGuard<'_, Vec<T>>> {
        self.0.public.read()
    }

    pub fn memory(&self) -> LockResult<RwLockReadGuard<'_, MemoryMap<T>>> {
        self.0.memory.read()
    }

    pub fn memory_mut(&self) -> LockResult<RwLockWriteGuard<'_, MemoryMap<T>>> {
        self.0.memory.write()
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

    /// Evaluates the log derivative entry `LogEntry` at the given row index.
    #[inline]
    pub fn read_log_entry<T: EvalCubic>(
        &self,
        entry: &LogEntry<T>,
        row_index: usize,
    ) -> LogEntryValue<F> {
        let eval = |value: &T| T::trace_value_as_cubic(self.read(value, row_index));
        match entry {
            LogEntry::Input(value) => LogEntryValue {
                value: eval(value),
                multiplier: F::ONE,
            },
            LogEntry::Output(value) => LogEntryValue {
                value: eval(value),
                multiplier: -F::ONE,
            },
            LogEntry::InputMultiplicity(value, multiplier) => {
                let multiplier = self.read(multiplier, row_index);
                LogEntryValue {
                    value: eval(value),
                    multiplier,
                }
            }
            LogEntry::OutputMultiplicity(value, multiplier) => {
                let multiplier = self.read(multiplier, row_index);
                LogEntryValue {
                    value: eval(value),
                    multiplier: -multiplier,
                }
            }
        }
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
    pub fn write_trace_slice<T: RegisterSerializable>(
        &self,
        data: &T,
        value: &[F],
        row_index: usize,
    ) {
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
    pub fn write_slice<T: RegisterSerializable>(&self, data: &T, value: &[F], row_index: usize) {
        let register = data.register();
        match register {
            MemorySlice::Local(..) => self.write_trace_slice(data, value, row_index),
            MemorySlice::Next(..) => self.write_trace_slice(data, value, row_index),
            MemorySlice::Global(..) => {
                let mut global = self.0.global.write().unwrap();
                register.assign_to_raw_slice(&mut global, value);
            }
            MemorySlice::Public(..) => {
                let mut public = self.0.public.write().unwrap();
                register.assign_to_raw_slice(&mut public, value);
            }
            MemorySlice::Challenge(..) => unreachable!("Challenge registers are read-only"),
        }
    }

    #[inline]
    pub fn write<T: Register>(&self, data: &T, value: &T::Value<F>, row_index: usize) {
        self.write_slice(data, T::align(value), row_index)
    }

    #[inline]
    pub fn write_instruction(&self, instruction: &impl Instruction<F>, row_index: usize) {
        instruction.write(self, row_index)
    }

    #[inline]
    pub fn write_row_instructions<L: AirParameters<Field = F>>(
        &self,
        air_data: &AirTraceData<L>,
        row_index: usize,
    ) {
        for instruction in air_data.instructions.iter() {
            self.write_instruction(instruction, row_index);
        }
    }

    #[inline]
    pub fn write_global_instructions<L: AirParameters<Field = F>>(
        &self,
        air_data: &AirTraceData<L>,
    ) {
        for instruction in air_data.global_instructions.iter() {
            self.write_instruction(instruction, 0);
        }
    }

    /// An atomic fetch and modify operation on a register.
    #[inline]
    pub fn fetch_and_modify<T: Register>(
        &self,
        data: &T,
        op: impl FnOnce(&T::Value<F>) -> T::Value<F>,
        row_index: usize,
    ) {
        match data.register() {
            MemorySlice::Local(..) => {
                let mut trace = self.0.trace.write().unwrap();
                let window = trace.window(row_index);
                let parser = TraceWindowParser::new(window, &[], &[], &[]);
                let value = data.eval(&parser);

                let new_value = op(&value);
                data.register()
                    .assign(&mut trace.view_mut(), 0, T::align(&new_value), row_index);
            }
            MemorySlice::Next(..) => {
                let mut trace = self.0.trace.write().unwrap();
                let window = trace.window(row_index);
                let parser = TraceWindowParser::new(window, &[], &[], &[]);
                let value = data.eval(&parser);

                let new_value = op(&value);
                data.register()
                    .assign(&mut trace.view_mut(), 0, T::align(&new_value), row_index);
            }
            MemorySlice::Global(..) => {
                let mut global = self.0.global.write().unwrap();
                let value = data.read_from_slice(&global);
                let new_value = op(&value);
                data.assign_to_raw_slice(&mut global, &new_value);
            }
            MemorySlice::Public(..) => {
                let mut public = self.0.public.write().unwrap();
                let value = data.read_from_slice(&public);
                let new_value = op(&value);
                data.assign_to_raw_slice(&mut public, &new_value);
            }
            MemorySlice::Challenge(..) => unreachable!("Challenge registers are read-only"),
        }
    }
}

impl<T: PartialEq + Eq + Hash> Deref for TraceWriter<T> {
    type Target = WriterData<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
