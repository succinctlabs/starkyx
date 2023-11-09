use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::trace::writer::AirWriter;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

pub mod assign;
pub mod bit;
pub mod cycle;
pub mod empty;
pub mod node;
pub mod set;

pub trait Instruction<F: Field>:
    'static + Send + Sync + Clone + Debug + Serialize + for<'de> Deserialize<'de>
{
    /// Writes the instruction to the trace.
    fn write(&self, writer: &TraceWriter<F>, row_index: usize);

    #[allow(unused_variables)]
    // Writes the instruction to a general AirWriter.
    fn write_to_air(&self, writer: &mut impl AirWriter<F>) {
        unimplemented!("write_to_air not implemented for this instruction")
    }
}

/// An instruction that only consists of constraints
pub trait ConstraintInstruction:
    'static + Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
}

impl<F: Field, C: ConstraintInstruction> Instruction<F> for C {
    fn write(&self, _writer: &TraceWriter<F>, _row_index: usize) {}
}
