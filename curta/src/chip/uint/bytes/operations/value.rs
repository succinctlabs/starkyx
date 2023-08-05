use super::{
    NUM_CHALLENGES, OPCODE_AND, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR, OPCODE_XOR,
};
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::bit_operations::util::u8_to_bits_le;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ByteOperation<T> {
    And(T, T, T),
    Xor(T, T, T),
    Shr(T, T, T),
    Rot(T, T, T),
    Not(T, T),
    Range(T),
}

impl ByteOperation<ByteRegister> {
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

    pub fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            ByteOperation::And(a, b, _) => vec![*a.register(), *b.register()],
            ByteOperation::Xor(a, b, _) => vec![*a.register(), *b.register()],
            ByteOperation::Shr(a, b, _) => vec![*a.register(), *b.register()],
            ByteOperation::Rot(a, b, _) => vec![*a.register(), *b.register()],
            ByteOperation::Not(a, _) => vec![*a.register()],
            ByteOperation::Range(a) => vec![*a.register()],
        }
    }

    pub fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            ByteOperation::And(_, _, c) => vec![*c.register()],
            ByteOperation::Xor(_, _, c) => vec![*c.register()],
            ByteOperation::Shr(_, _, c) => vec![*c.register()],
            ByteOperation::Rot(_, _, c) => vec![*c.register()],
            ByteOperation::Not(_, b) => vec![*b.register()],
            ByteOperation::Range(_) => vec![],
        }
    }

    pub fn write<F: PrimeField64>(
        &self,
        writer: &TraceWriter<F>,
        row_index: usize,
    ) -> ByteOperation<u8> {
        let from_field = |x: F| F::as_canonical_u64(&x) as u8;
        let as_field = |x| F::from_canonical_u8(x);
        match self {
            ByteOperation::And(a, b, c) => {
                let a_val = from_field(writer.read(a, row_index));
                let b_val = from_field(writer.read(b, row_index));
                let c_val = a_val & b_val;
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::And(a_val, b_val, c_val)
            }
            ByteOperation::Xor(a, b, c) => {
                let a_val = from_field(writer.read(a, row_index));
                let b_val = from_field(writer.read(b, row_index));
                let c_val = a_val ^ b_val;
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::Xor(a_val, b_val, c_val)
            }
            ByteOperation::Shr(a, b, c) => {
                let a_val = from_field(writer.read(a, row_index));
                let b_val = from_field(writer.read(b, row_index));
                let c_val = a_val >> b_val;
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::Shr(a_val, b_val, c_val)
            }
            ByteOperation::Rot(a, b, c) => {
                let a_val = from_field(writer.read(a, row_index));
                let b_val = from_field(writer.read(b, row_index));
                let c_val = a_val.rotate_right(b_val as u32);
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::Rot(a_val, b_val, c_val)
            }
            ByteOperation::Not(a, b) => {
                let a_val = from_field(writer.read(a, row_index));
                let b_val = !a_val;
                writer.write(b, &as_field(b_val), row_index);
                ByteOperation::Not(a_val, b_val)
            }
            ByteOperation::Range(a) => {
                let a_val = from_field(writer.read(a, row_index));
                ByteOperation::Range(a_val)
            }
        }
    }
}

impl<T> ByteOperation<T> {
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

    pub fn from_opcode_and_values(opcode: u32, a: T, b: T, c: Option<T>) -> Self {
        match opcode {
            OPCODE_AND => ByteOperation::And(a, b, c.unwrap()),
            OPCODE_XOR => ByteOperation::Xor(a, b, c.unwrap()),
            OPCODE_SHR => ByteOperation::Shr(a, b, c.unwrap()),
            OPCODE_ROT => ByteOperation::Rot(a, b, c.unwrap()),
            OPCODE_NOT => ByteOperation::Not(a, c.unwrap()),
            OPCODE_RANGE => ByteOperation::Range(a),
            _ => panic!("Invalid opcode {}", opcode),
        }
    }

    pub fn field_opcode<F: Field>(&self) -> F {
        F::from_canonical_u32(self.opcode())
    }
}

impl ByteOperation<u8> {
    pub fn and(a: u8, b: u8) -> Self {
        ByteOperation::And(a, b, a & b)
    }

    pub fn xor(a: u8, b: u8) -> Self {
        ByteOperation::Xor(a, b, a ^ b)
    }

    pub fn shr(a: u8, b: u8) -> Self {
        ByteOperation::Shr(a, b, a >> b)
    }

    pub fn rot(a: u8, b: u8) -> Self {
        ByteOperation::Rot(a, b, a.rotate_right(b as u32))
    }

    pub fn not(a: u8) -> Self {
        ByteOperation::Not(a, !a)
    }

    pub fn range(a: u8) -> Self {
        ByteOperation::Range(a)
    }

    pub fn as_field_op<F: Field>(self) -> ByteOperation<F> {
        let as_field = |x| F::from_canonical_u8(x);
        match self {
            ByteOperation::And(a, b, c) => {
                ByteOperation::And(as_field(a), as_field(b), as_field(c))
            }
            ByteOperation::Xor(a, b, c) => {
                ByteOperation::Xor(as_field(a), as_field(b), as_field(c))
            }
            ByteOperation::Shr(a, b, c) => {
                ByteOperation::Shr(as_field(a), as_field(b), as_field(c))
            }
            ByteOperation::Rot(a, b, c) => {
                ByteOperation::Rot(as_field(a), as_field(b), as_field(c))
            }
            ByteOperation::Not(a, b) => ByteOperation::Not(as_field(a), as_field(b)),
            ByteOperation::Range(a) => ByteOperation::Range(as_field(a)),
        }
    }

    pub fn as_field_bits_op<F: Field>(self) -> ByteOperation<[F; 8]> {
        let as_field_bits = |x| u8_to_bits_le(x).map(|b| F::from_canonical_u8(b));
        match self {
            ByteOperation::And(a, b, c) => {
                ByteOperation::And(as_field_bits(a), as_field_bits(b), as_field_bits(c))
            }
            ByteOperation::Xor(a, b, c) => {
                ByteOperation::Xor(as_field_bits(a), as_field_bits(b), as_field_bits(c))
            }
            ByteOperation::Shr(a, b, c) => {
                ByteOperation::Shr(as_field_bits(a), as_field_bits(b), as_field_bits(c))
            }
            ByteOperation::Rot(a, b, c) => {
                ByteOperation::Rot(as_field_bits(a), as_field_bits(b), as_field_bits(c))
            }
            ByteOperation::Not(a, b) => ByteOperation::Not(as_field_bits(a), as_field_bits(b)),
            ByteOperation::Range(a) => ByteOperation::Range(as_field_bits(a)),
        }
    }
}
