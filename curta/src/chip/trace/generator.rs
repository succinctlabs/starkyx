use anyhow::{Error, Result};

use crate::chip::{Chip, AirParameters};
use crate::math::prelude::*;
use crate::trace::generator::TraceGenerator;
use crate::trace::AirTrace;

#[derive(Debug, Clone)]
pub struct CurtaGenerator<L : AirParameters> {
    pub trace: Option<AirTrace<L::Field>>,
}

impl<L : AirParameters> TraceGenerator<L::Field, Chip<L>> for CurtaGenerator<L> {
    type Error = Error;

    fn generate_round(
            &self,
            air: &Chip<L>,
            round: usize,
            challenges: &[L::Field],
        ) -> Result<AirTrace<L::Field>> {
        todo!()
    }
}