use super::lookup_table::NUM_CHALLENGES;
use super::register::ByteRegister;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::Register;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

impl ByteOperation {

    pub const fn opcode(&self) -> u32 {
        match self {
            ByteOperation::And(_, _, _) => OPCODE_AND,
            ByteOperation::Xor(_, _, _) => OPCODE_XOR,
            ByteOperation::Adc(_, _, _, _, _) => OPCODE_ADC,
            ByteOperation::Shr(_, _, _, _) => OPCODE_SHR,
            ByteOperation::Shl(_, _, _, _) => OPCODE_SHL,
            ByteOperation::Not(_, _) => OPCODE_NOT,
        }
    }

    pub fn field_opcode<F: Field>(&self) -> F {
        F::from_canonical_u32(self.opcode())
    }

    pub fn expression_array<F: Field>(&self)  -> [ArithmeticExpression<F>; NUM_CHALLENGES] {
        match self {
            ByteOperation::And(a, b, c) => {
                [
                    ArithmeticExpression::from(self.field_opcode::<F>()),
                    a.expr(),
                    b.expr(),
                    c.expr(),
                    ArithmeticExpression::zero(),
                    ArithmeticExpression::zero(),
                ]
            },
            ByteOperation::Xor(a, b, c) => {
                [
                    ArithmeticExpression::from(self.field_opcode::<F>()),
                    a.expr(),
                    b.expr(),
                    c.expr(),
                    ArithmeticExpression::zero(),
                    ArithmeticExpression::zero(),
                ]
            },
            ByteOperation::Adc(a, b, c, d, e) => {
                [
                    ArithmeticExpression::from(self.field_opcode::<F>()),
                    a.expr(),
                    b.expr(),
                    c.expr(),
                    d.expr(),
                    e.expr(),
                ]
            },
            ByteOperation::Shr(a, b, c, d) => {
                [
                    ArithmeticExpression::from(self.field_opcode::<F>()),
                    a.expr(),
                    b.expr(),
                    c.expr(),
                    d.expr(),
                    ArithmeticExpression::zero(),
                ]
            },
            ByteOperation::Shl(a, b, c, d) => {
                [
                    ArithmeticExpression::from(self.field_opcode::<F>()),
                    a.expr(),
                    b.expr(),
                    c.expr(),
                    d.expr(),
                    ArithmeticExpression::zero(),
                ]
            },
            ByteOperation::Not(a, b) => {
                [
                    ArithmeticExpression::from(self.field_opcode::<F>()),
                    a.expr(),
                    b.expr(),
                    ArithmeticExpression::zero(),
                    ArithmeticExpression::zero(),
                    ArithmeticExpression::zero(),
                ]
            },
        } 
    }
}