use super::{OPCODE_AND, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR, OPCODE_XOR};
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ByteOperationValue<T> {
    And(T, T, T),
    Xor(T, T, T),
    Shr(T, T, T),
    Rot(T, T, T),
    Not(T, T),
    Range(T),
}

impl<T> ByteOperationValue<T> {
    pub fn opcode(&self) -> u32 {
        match self {
            ByteOperationValue::And(_, _, _) => OPCODE_AND,
            ByteOperationValue::Xor(_, _, _) => OPCODE_XOR,
            ByteOperationValue::Shr(_, _, _) => OPCODE_SHR,
            ByteOperationValue::Rot(_, _, _) => OPCODE_ROT,
            ByteOperationValue::Not(_, _) => OPCODE_NOT,
            ByteOperationValue::Range(_) => OPCODE_RANGE,
        }
    }
}

impl ByteOperationValue<u8> {
    pub fn and(a: u8, b: u8) -> Self {
        ByteOperationValue::And(a, b, a & b)
    }

    pub fn xor(a: u8, b: u8) -> Self {
        ByteOperationValue::Xor(a, b, a ^ b)
    }

    pub fn shr(a: u8, b: u8) -> Self {
        ByteOperationValue::Shr(a, b, a >> b)
    }

    pub fn rot(a: u8, b: u8) -> Self {
        ByteOperationValue::Rot(a, b, a.rotate_right(b as u32))
    }

    pub fn not(a: u8) -> Self {
        ByteOperationValue::Not(a, !a)
    }

    pub fn range(a: u8) -> Self {
        ByteOperationValue::Range(a)
    }

    pub fn as_field_op<F: Field>(self) -> ByteOperationValue<F> {
        let as_field = |x| F::from_canonical_u8(x);
        match self {
            ByteOperationValue::And(a, b, c) => {
                ByteOperationValue::And(as_field(a), as_field(b), as_field(c))
            }
            ByteOperationValue::Xor(a, b, c) => {
                ByteOperationValue::Xor(as_field(a), as_field(b), as_field(c))
            }
            ByteOperationValue::Shr(a, b, c) => {
                ByteOperationValue::Shr(as_field(a), as_field(b), as_field(c))
            }
            ByteOperationValue::Rot(a, b, c) => {
                ByteOperationValue::Rot(as_field(a), as_field(b), as_field(c))
            }
            ByteOperationValue::Not(a, b) => ByteOperationValue::Not(as_field(a), as_field(b)),
            ByteOperationValue::Range(a) => ByteOperationValue::Range(as_field(a)),
        }
    }

    pub fn from_field_op<F: PrimeField64>(op: &ByteOperationValue<F>) -> Self {
        let from_field = |x: &F| F::as_canonical_u64(x) as u8;

        match op {
            ByteOperationValue::And(a, b, c) => {
                ByteOperationValue::And(from_field(a), from_field(b), from_field(c))
            }
            ByteOperationValue::Xor(a, b, c) => {
                ByteOperationValue::Xor(from_field(a), from_field(b), from_field(c))
            }
            ByteOperationValue::Shr(a, b, c) => {
                ByteOperationValue::Shr(from_field(a), from_field(b), from_field(c))
            }
            ByteOperationValue::Rot(a, b, c) => {
                ByteOperationValue::Rot(from_field(a), from_field(b), from_field(c))
            }
            ByteOperationValue::Not(a, b) => ByteOperationValue::Not(from_field(a), from_field(b)),
            ByteOperationValue::Range(a) => ByteOperationValue::Range(from_field(a)),
        }
    }
}
