#![allow(dead_code)]

pub mod arithmetic_stark;
pub mod builder;
pub mod chip;
pub mod ec;
pub mod eddsa;
pub mod field;
pub mod instruction;
pub mod layout;
pub mod modular;
pub mod polynomial;
pub mod register;
pub mod trace;
pub(crate) mod util;

use std::sync::mpsc::Sender;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
pub use register::Register;

use crate::arithmetic::chip::EmulatedCircuitLayout;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub trait InstructionT<
    L: EmulatedCircuitLayout<F, D, N>,
    F: RichField + Extendable<D>,
    const D: usize,
    const N: usize,
>: 'static + Sized + Send + Sync
{
    fn generate_trace(self, pc: usize, tx: Sender<(usize, usize, Vec<F>)>);
}

/// An experimental parser to generate Stark constaint code from commands
///
/// The output is writing to a "memory" passed to it.
#[derive(Debug, Clone, Copy)]
pub struct ArithmeticParser<F, const D: usize> {
    _marker: core::marker::PhantomData<F>,
}
