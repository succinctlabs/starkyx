pub mod and;
pub mod or;
pub mod rotate;
pub mod shr;
pub mod xor;

use and::And;

use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;

#[derive(Debug, Clone, Copy)]
pub enum BitOperation<const NUM_BITS: usize> {
    And(And<NUM_BITS>),
}

#[derive(Debug, Clone)]
pub struct U32BitOperation {
    pub a: ElementRegister,
    pub b: ElementRegister,
    pub output: ElementRegister,
    pub digest: CubicRegister,
    pub operation: BitOperation<32>,
}
