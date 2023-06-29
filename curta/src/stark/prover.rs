use super::config::StarkConfig;
use super::Stark;
use crate::air::parser::AirParser;

pub trait StarkProver<AP: AirParser, C: StarkConfig<AP>> {
    /// The error type returned by the prover.
    type Error;

    /// Prove a batch of STARKs.
    fn prove_batch<S: Stark<AP, C>>(&self, stark: &[&S]) -> Result<C::Proof, Self::Error>;

    /// Prove a single STARK.
    fn prove<S: Stark<AP, C>>(&self, stark: &S) -> Result<C::Proof, Self::Error> {
        self.prove_batch(&[stark])
    }
}
