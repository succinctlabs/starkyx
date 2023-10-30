//! Byte operations with a lookup table

use self::instruction::ByteOperationInstruction;
use self::value::ByteOperation;
use super::lookup_table::builder_operations::ByteLookupOperations;
use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::register::element::ElementRegister;
use crate::chip::AirParameters;

pub mod instruction;
pub mod value;

pub const OPCODE_AND: u8 = 101;
pub const OPCODE_XOR: u8 = 102;
pub const OPCODE_SHR: u8 = 103;
pub const OPCODE_ROT: u8 = 104;
pub const OPCODE_NOT: u8 = 105;
pub const OPCODE_RANGE: u8 = 106;

pub const NUM_BIT_OPPS: usize = 5;

pub const OPCODE_INDICES: [u8; NUM_BIT_OPPS + 1] = [
    OPCODE_AND,
    OPCODE_XOR,
    OPCODE_SHR,
    OPCODE_ROT,
    OPCODE_NOT,
    OPCODE_RANGE,
];

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_byte_operation(
        &mut self,
        op: &ByteOperation<ByteRegister>,
        lookup_values: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let digest = self.alloc::<ElementRegister>();

        let instr = ByteOperationInstruction::new(*op, digest, false);
        lookup_values.values.push(digest);
        self.register_instruction(instr);
    }

    pub fn set_public_inputs_byte_operation(
        &mut self,
        op: &ByteOperation<ByteRegister>,
        lookup_values: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        // TODO: assert that the inputs are public (for debugging purpuoses)

        let digest = self.alloc_public::<ElementRegister>();

        let instr = ByteOperationInstruction::new(*op, digest, true);
        lookup_values.values.push(digest);
        self.register_global_instruction(instr);
    }
}
