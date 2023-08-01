use super::bit_operations::BitOperation;
use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::AirParameters;

pub const OPCODE_AND: u32 = 97;
pub const OPCODE_OR: u32 = 101;
pub const OPCODE_XOR: u32 = 102;
pub const OPCODE_ADD: u32 = 103;
pub const fn opcode_rotate(n: usize) -> u32 {
    4 + ((n as u32) << 8u32)
}
pub const fn opcode_shr(n: usize) -> u32 {
    5 + ((n as u32) << 8u32)
}

#[derive(Debug, Clone, Copy)]
pub struct ByteLookup<const NUM_OPS: usize> {
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub results: ArrayRegister<ByteRegister>,
    a_bits: ArrayRegister<BitRegister>,
    b_bits: ArrayRegister<BitRegister>,
    results_bits: [ArrayRegister<BitRegister>; 8],
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn bytes_lookup(&mut self) -> ByteLookup<8> {
        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();
        let results = self.alloc_array::<ByteRegister>(8);
        let a_bits = self.alloc_array::<BitRegister>(8);
        let b_bits = self.alloc_array::<BitRegister>(8);
        let results_bits = [self.alloc_array::<BitRegister>(8); 8];
        ByteLookup {
            a,
            b,
            results,
            a_bits,
            b_bits,
            results_bits,
        }
    }
}
