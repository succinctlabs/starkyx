//! Byte operations with a lookup table

pub mod instruction;
pub mod value;

pub const OPCODE_AND: u32 = 101;
pub const OPCODE_XOR: u32 = 102;
pub const OPCODE_SHR: u32 = 103;
pub const OPCODE_ROT: u32 = 104;
pub const OPCODE_NOT: u32 = 105;
pub const OPCODE_RANGE: u32 = 106;

pub const NUM_BIT_OPPS: usize = 5;

pub const OPCODE_INDICES: [u32; NUM_BIT_OPPS + 1] = [
    OPCODE_AND,
    OPCODE_XOR,
    OPCODE_SHR,
    OPCODE_ROT,
    OPCODE_NOT,
    OPCODE_RANGE,
];

pub const NUM_CHALLENGES: usize = 4;
