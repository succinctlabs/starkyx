use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::AirParameters;

pub const OPCODE_AND: u32 = 101;
pub const OPCODE_XOR: u32 = 102;
pub const OPCODE_ADC: u32 = 103;
pub const OPCODE_SHR: u32 = 104;
pub const OPCODE_SHL: u32 = 105;
pub const OPCODE_NOT: u32 = 106;

#[derive(Debug, Clone, Copy)]
pub struct ByteLookup<const NUM_OPS: usize> {
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub results: ArrayRegister<ByteRegister>,
    a_bits: ArrayRegister<BitRegister>,
    b_bits: ArrayRegister<BitRegister>,
    results_bits: [ArrayRegister<BitRegister>; NUM_OPS],
    carry_bits: [BitRegister; NUM_OPS],
}

impl<L: AirParameters> AirBuilder<L> {
    pub const NUM_BIT_OPS: usize = 6;

    pub fn bytes_lookup(&mut self) -> ByteLookup<{ Self::NUM_BIT_OPS }> {
        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();
        let results = self.alloc_array::<ByteRegister>(8);
        let a_bits = self.alloc_array::<BitRegister>(8);
        let b_bits = self.alloc_array::<BitRegister>(8);
        let results_bits = [self.alloc_array::<BitRegister>(8); { Self::NUM_BIT_OPS }];
        let carry_bits = [self.alloc::<BitRegister>(); { Self::NUM_BIT_OPS }];
        ByteLookup {
            a,
            b,
            results,
            a_bits,
            b_bits,
            results_bits,
            carry_bits,
        }
    }
}
