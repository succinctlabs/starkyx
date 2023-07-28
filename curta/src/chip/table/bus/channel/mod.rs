pub mod constraint;
pub mod entry;
pub mod trace;

use core::marker::PhantomData;

use self::entry::Entry;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::cubic::CubicRegister;
use crate::math::prelude::*;

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct BusChannel<F, E> {
    pub out_channel: CubicRegister,
    table_accumulator: CubicRegister,
    challenge: CubicRegister,
    entries: Vec<Entry<F>>,
    _marker: PhantomData<E>,
}

impl<F: Field, E> BusChannel<F, E> {
    pub fn new(
        challenge: CubicRegister,
        out_channel: CubicRegister,
        table_accumulator: CubicRegister,
    ) -> Self {
        Self {
            challenge,
            out_channel,
            table_accumulator,
            entries: Vec::new(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn input(&mut self, register: CubicRegister) {
        self.input_filtered(register, ArithmeticExpression::one());
    }

    #[inline]
    pub fn input_filtered(&mut self, register: CubicRegister, filter: ArithmeticExpression<F>) {
        self.entries.push(Entry::Input(register, filter));
    }

    #[inline]
    pub fn output(&mut self, register: CubicRegister) {
        self.output_filtered(register, ArithmeticExpression::one())
    }

    #[inline]
    pub fn output_filtered(&mut self, register: CubicRegister, filter: ArithmeticExpression<F>) {
        self.entries.push(Entry::Output(register, filter));
    }
}
