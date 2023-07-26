pub mod and;
pub mod or;
pub mod rotate;
pub mod shr;

use and::And;

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::extension::ExtensionRegister;

#[derive(Debug, Clone, Copy)]
pub enum BitOperation<const NUM_BITS: usize> {
    And(And<NUM_BITS>),
}

#[derive(Debug, Clone)]
pub struct U32BitOperation {
    pub a: ElementRegister,
    pub b: ElementRegister,
    pub output: ElementRegister,
    pub digest: ExtensionRegister<3>,
    pub digest_challenges: ArrayRegister<ExtensionRegister<3>>,
    operation: BitOperation<32>,
}
