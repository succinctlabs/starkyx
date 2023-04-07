#![allow(dead_code)]

pub mod arithmetic_stark;
pub mod eddsa;
pub mod polynomial;
pub(crate) mod util;
pub mod circuit;
pub mod instruction;
pub mod builder;
pub mod modular;
pub mod layout;
pub mod register;
pub mod trace;

pub use register::Register;

use std::sync::mpsc::Sender;

use plonky2::field::extension::{Extendable};
use plonky2::hash::hash_types::RichField;

use crate::arithmetic::circuit::EmulatedCircuitLayout;
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
