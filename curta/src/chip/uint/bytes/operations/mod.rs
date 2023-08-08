//! Byte operations with a lookup table

use self::instruction::ByteOperationInstruction;
use self::value::ByteOperation;
use super::lookup_table::builder_operations::ByteLookupOperations;
use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::AirParameters;

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

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_byte_operation(
        &mut self,
        op: &ByteOperation<ByteRegister>,
        lookup_values: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let tx = lookup_values.tx.clone();

        let digest =
            self.accumulate_expressions(&lookup_values.row_acc_challenges, &op.expression_array());

        let instr = ByteOperationInstruction::new(tx, *op, false);
        self.register_instruction(instr);
        lookup_values.values.push(digest);
    }

    pub fn set_public_inputs_byte_operation(
        &mut self,
        op: &ByteOperation<ByteRegister>,
        lookup_values: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        // TODO: Check that the inputs are public
        let tx = lookup_values.tx.clone();

        let digest =
            self.accumulate_public_expressions(&lookup_values.row_acc_challenges, &op.expression_array());

        let instr = ByteOperationInstruction::new(tx, *op, true);
        self.register_instruction(instr);
        lookup_values.values.push(digest);
    }
}
