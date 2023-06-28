use self::config::StarkConfig;

pub mod config;
pub mod proof;
pub mod verifier;
pub mod prover;


pub trait Stark<C: StarkConfig> {
}