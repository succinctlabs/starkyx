use crate::math::prelude::*;

pub trait TraceCS<F: Field> {
    type Commitment;

    type Proof;
    type ProverData;
    type VerifierData;

    fn commit(&self) -> Self::ProverData;
}


pub trait TCSProver<F: Field, TCS: TraceCS<F>> {}
