use crate::chip::register::{bit::BitRegister, cubic::CubicRegister};

use super::{bit_operations::BitOperation, register::ByteRegister};




#[derive(Debug, Clone, Copy)]
pub struct ByteOperations {
    pub a : ByteRegister,
    pub b : ByteRegister,
    pub result : ByteRegister,
    pub bit_operation : BitOperation<8>,
}

#[derive(Debug, Clone, Copy)]
pub struct ByteLookup {
    and_digest : CubicRegister,
    and_multiplicity: usize,
    xor_digest : CubicRegister,
    xor_multiplicity: usize,
    not_digest : CubicRegister,
    not_multiplicity: usize,
    adc_digest : CubicRegister,
    adc_multiplicity: usize,
    or_digest : CubicRegister,
    or_multiplicity: usize,
    rotate_right_digest : CubicRegister,
    rotate_right_multiplicity: usize,
    shr_digest : CubicRegister,
    shr_multiplicity: usize,
    shl_digest : CubicRegister,
    shl_multiplicity: usize,
}