use anyhow::{anyhow, Error};

use super::StarkTrace;
use crate::air::parser::AirParser;
use crate::stark::config::StarkConfig;
use crate::stark::Stark;

pub trait TraceGenerator<AP: AirParser, C: StarkConfig<AP>> {
    type Error;
    fn generate_round<S: Stark<AP, C>>(
        &self,
        stark: &S,
        round: usize,
        challenges: &[AP::Field],
    ) -> Result<StarkTrace<AP::Field>, Self::Error>;
}

#[derive(Debug, Clone)]
pub struct ConstantGenerator<AP: AirParser> {
    trace: StarkTrace<AP::Field>,
}

impl<AP: AirParser> ConstantGenerator<AP> {
    pub fn new(trace: StarkTrace<AP::Field>) -> Self {
        Self { trace }
    }
}

impl<AP: AirParser, C: StarkConfig<AP>> TraceGenerator<AP, C> for ConstantGenerator<AP> {
    type Error = Error;

    fn generate_round<S: Stark<AP, C>>(
        &self,
        _stark: &S,
        round: usize,
        _challenges: &[AP::Field],
    ) -> Result<StarkTrace<AP::Field>, Self::Error> {
        match round {
            0 => Ok(self.trace.clone()),
            _ => Err(anyhow!("Constant generator has only one round")),
        }
    }
}
