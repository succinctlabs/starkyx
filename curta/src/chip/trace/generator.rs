use core::marker::PhantomData;
use std::sync::RwLock;

use alloc::sync::Arc;
use anyhow::{Error, Result};

use crate::chip::AirParameters;
use crate::maybe_rayon::*;
use crate::trace::AirTrace;

use super::writer::TraceWriter;

#[derive(Debug)]
pub struct ArithmeticGenerator<L: AirParameters>{
    writer: TraceWriter<L::Field>,
}

impl<L : ~const AirParameters> ArithmeticGenerator<L> {
    pub fn new() -> Self {
        Self {
            writer : TraceWriter::new(L::num_columns(), L::num_rows()),
        }
    }
}