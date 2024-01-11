//! Byte operations with a lookup table

use self::instruction::ByteOperationInstruction;
use self::value::ByteOperation;
use super::lookup_table::builder_operations::ByteLookupOperations;
use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::AirParameters;

pub mod instruction;
pub mod value;

pub const OPCODE_AND: u8 = 101;
pub const OPCODE_XOR: u8 = 102;
pub const OPCODE_SHR: u8 = 103;
pub const OPCODE_ROT: u8 = 104;
pub const OPCODE_NOT: u8 = 105;
pub const OPCODE_RANGE: u8 = 106;
pub const OPCODE_SHR_CARRY: u8 = 107;

pub const NUM_BIT_OPPS: usize = 6;

pub const OPCODE_INDICES: [u8; NUM_BIT_OPPS + 1] = [
    OPCODE_AND,
    OPCODE_XOR,
    OPCODE_SHR,
    OPCODE_SHR_CARRY,
    OPCODE_ROT,
    OPCODE_NOT,
    OPCODE_RANGE,
];

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_byte_operation(
        &mut self,
        op: &ByteOperation<ByteRegister>,
        lookup: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let instr = ByteOperationInstruction::new(*op, false);
        lookup.trace_operations.push(*op);
        self.register_instruction(instr);
    }

    pub fn set_public_inputs_byte_operation(
        &mut self,
        op: &ByteOperation<ByteRegister>,
        lookup: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let instr = ByteOperationInstruction::new(*op, true);
        lookup.public_operations.push(*op);
        self.register_global_instruction(instr);
    }
}
