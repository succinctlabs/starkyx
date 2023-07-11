use alloc::sync::Arc;
use core::marker::PhantomData;
use std::sync::RwLock;

use anyhow::{Error, Result};

use super::writer::TraceWriter;
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::trace::generator::TraceGenerator;
use crate::trace::AirTrace;

#[derive(Debug)]
pub struct ArithmeticGenerator<L: AirParameters> {
    writer: TraceWriter<L::Field>,
}

impl<L: ~const AirParameters> ArithmeticGenerator<L> {
    pub fn new() -> Self {
        Self {
            writer: TraceWriter::new_with_value(L::num_columns(), L::num_rows(), L::Field::ZERO),
        }
    }

    pub fn new_writer(&self) -> TraceWriter<L::Field> {
        self.writer.clone()
    }

    pub fn trace(&self) -> AirTrace<L::Field> {
        self.writer.trace().clone()
    }
}

impl<L: AirParameters> TraceGenerator<L::Field, Chip<L>> for ArithmeticGenerator<L> {
    type Error = Error;

    fn generate_round(
        &self,
        air: &Chip<L>,
        round: usize,
        challenges: &[L::Field],
        public_inputs: &[L::Field],
    ) -> Result<AirTrace<L::Field>> {
        match round {
            0 => Ok(self.trace()),
            _ => todo!("Implement me"),
        }
    }
}
