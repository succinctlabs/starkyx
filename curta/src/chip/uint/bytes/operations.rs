use super::bit_operations::BitOperation;
use super::register::ByteRegister;
use crate::chip::register::cubic::CubicRegister;

#[derive(Debug, Clone, Copy)]
pub struct ByteOperations {
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub result: ByteRegister,
    pub bit_operation: BitOperation<8>,
}

#[derive(Debug, Clone, Copy)]
pub struct ByteLookup {}
