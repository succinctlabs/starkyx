pub mod instruction;

use super::register::ByteRegister;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::Register;
use crate::math::prelude::*;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ByteOperation {
    And(ByteRegister, ByteRegister, ByteRegister),
    Xor(ByteRegister, ByteRegister, ByteRegister),
    Shr(ByteRegister, ByteRegister, ByteRegister),
    Rot(ByteRegister, ByteRegister, ByteRegister),
    Not(ByteRegister, ByteRegister),
    Range(ByteRegister),
}

impl ByteOperation {
    pub const fn opcode(&self) -> u32 {
        match self {
            ByteOperation::And(_, _, _) => OPCODE_AND,
            ByteOperation::Xor(_, _, _) => OPCODE_XOR,
            ByteOperation::Shr(_, _, _) => OPCODE_SHR,
            ByteOperation::Rot(_, _, _) => OPCODE_ROT,
            ByteOperation::Not(_, _) => OPCODE_NOT,
            ByteOperation::Range(_) => OPCODE_RANGE,
        }
    }

    pub fn field_opcode<F: Field>(&self) -> F {
        F::from_canonical_u32(self.opcode())
    }

    pub fn expression_array<F: Field>(&self) -> [ArithmeticExpression<F>; NUM_CHALLENGES] {
        let opcode = ArithmeticExpression::from(self.field_opcode::<F>());
        match self {
            ByteOperation::And(a, b, c) => [opcode, a.expr(), b.expr(), c.expr()],
            ByteOperation::Xor(a, b, c) => [opcode, a.expr(), b.expr(), c.expr()],
            ByteOperation::Shr(a, b, c) => [opcode, a.expr(), b.expr(), c.expr()],
            ByteOperation::Rot(a, b, c) => [opcode, a.expr(), b.expr(), c.expr()],
            ByteOperation::Not(a, b) => [opcode, a.expr(), b.expr(), ArithmeticExpression::zero()],
            ByteOperation::Range(a) => [
                opcode,
                a.expr(),
                ArithmeticExpression::zero(),
                ArithmeticExpression::zero(),
            ],
        }
    }
}
