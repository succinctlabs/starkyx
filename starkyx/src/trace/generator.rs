use core::fmt::Debug;

use anyhow::{anyhow, Error};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::AirTrace;
use crate::math::prelude::*;

pub trait TraceGenerator<F, A>:
    'static + Debug + Send + Sync + Serialize + DeserializeOwned
{
    type Error;
    fn generate_round(
        &self,
        air: &A,
        round: usize,
        challenges: &[F],
        global_values: &mut [F],
        public_inputs: &[F],
    ) -> Result<AirTrace<F>, Self::Error>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ConstantGenerator<F: Field> {
    trace: AirTrace<F>,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn new(trace: AirTrace<F>) -> Self {
        Self { trace }
    }
}

impl<F: Field, A> TraceGenerator<F, A> for ConstantGenerator<F> {
    type Error = Error;

    fn generate_round(
        &self,
        _air: &A,
        round: usize,
        _challenges: &[F],
        _global_values: &mut [F],
        _public_inputs: &[F],
    ) -> Result<AirTrace<F>, Self::Error> {
        match round {
            0 => Ok(self.trace.clone()),
            _ => Err(anyhow!("Constant generator has only one round")),
        }
    }
}
