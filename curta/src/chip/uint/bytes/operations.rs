use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::RegisterSerializable;
use crate::chip::table::lookup::log_der::{LogLookupValues, LookupTable};
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub const OPCODE_AND: u32 = 101;
pub const OPCODE_XOR: u32 = 102;
pub const OPCODE_ADC: u32 = 103;
pub const OPCODE_SHR: u32 = 104;
pub const OPCODE_SHL: u32 = 105;
pub const OPCODE_NOT: u32 = 106;

pub const NUM_BIT_OPPS: usize = 6;

pub const OPCODE_VALUES: [u32; NUM_BIT_OPPS] = [
    OPCODE_AND, OPCODE_XOR, OPCODE_ADC, OPCODE_SHR, OPCODE_SHL, OPCODE_NOT,
];

#[derive(Debug, Clone, Copy)]
pub enum ByteOperation {
    And(ByteRegister, ByteRegister, ByteRegister),
    Xor(ByteRegister, ByteRegister, ByteRegister),
    Adc(
        ByteRegister,
        ByteRegister,
        BitRegister,
        ByteRegister,
        BitRegister,
    ),
    Shr(ByteRegister, ByteRegister, ByteRegister, BitRegister),
    Shl(ByteRegister, ByteRegister, ByteRegister, BitRegister),
    Not(ByteRegister, ByteRegister),
}

// impl<F: Field> Instruction<F> for ByteOperation {
//     fn trace_layout(&self) -> Vec<MemorySlice> {
//         match self {
//             ByteOperation::And(a, b, c) => vec![*c.register()],
//         }
//     }
// }
