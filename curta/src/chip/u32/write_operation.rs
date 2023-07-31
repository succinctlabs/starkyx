use super::opcode::U32Opcode;
use super::operation::U32Operation;

#[derive(Debug, Clone)]
pub struct U32OperationWrite {
    pub opcode: U32Opcode,
    pub operation: U32Operation,
}
