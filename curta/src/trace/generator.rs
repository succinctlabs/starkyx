use anyhow::{anyhow, Error};

use super::StarkTrace;
use crate::air::parser::AirParser;
use crate::air::RAir;
use crate::math::prelude::*;

pub trait TraceGenerator<AP: AirParser> {
    type Error;
    fn generate_round<A: RAir<AP>>(
        &self,
        air: &A,
        round: usize,
        challenges: &[AP::Field],
    ) -> Result<StarkTrace<AP::Field>, Self::Error>;
}

#[derive(Debug, Clone)]
pub struct ConstantGenerator<F: Field> {
    trace: StarkTrace<F>,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn new(trace: StarkTrace<F>) -> Self {
        Self { trace }
    }
}

impl<F: Field, AP: AirParser<Field = F>> TraceGenerator<AP> for ConstantGenerator<F> {
    type Error = Error;

    fn generate_round<A: RAir<AP>>(
        &self,
        _air: &A,
        round: usize,
        _challenges: &[AP::Field],
    ) -> Result<StarkTrace<AP::Field>, Self::Error> {
        match round {
            0 => Ok(self.trace.clone()),
            _ => Err(anyhow!("Constant generator has only one round")),
        }
    }
}
