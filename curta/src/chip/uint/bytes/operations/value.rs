use serde::{Deserialize, Serialize};

use super::{
    NUM_CHALLENGES, OPCODE_AND, OPCODE_NOT, OPCODE_RANGE, OPCODE_ROT, OPCODE_SHR, OPCODE_XOR,
};
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::bit_operations::util::u8_to_bits_le;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ByteOperation<T> {
    And(T, T, T),
    Xor(T, T, T),
    Shr(T, T, T),
    ShrConst(T, u8, T),
    ShrCarry(T, u8, T, T),
    RotConst(T, u8, T),
    Rot(T, T, T),
    Not(T, T),
    Range(T),
}

impl ByteOperation<ByteRegister> {
    pub fn expression_array<F: Field>(&self) -> [ArithmeticExpression<F>; NUM_CHALLENGES] {
        let opcode = ArithmeticExpression::from(self.field_opcode::<F>());
        match self {
            ByteOperation::And(a, b, result) => [opcode, a.expr(), b.expr(), result.expr()],
            ByteOperation::Xor(a, b, result) => [opcode, a.expr(), b.expr(), result.expr()],
            ByteOperation::Shr(a, b, c) => [opcode, a.expr(), b.expr(), c.expr()],
            ByteOperation::ShrConst(a, b, c) => [
                opcode,
                a.expr(),
                ArithmeticExpression::from_constant(F::from_canonical_u8(*b)),
                c.expr(),
            ],
            ByteOperation::ShrCarry(a, shift, result, carry) => [
                opcode,
                a.expr(),
                ArithmeticExpression::from_constant(F::from_canonical_u8(*shift)),
                result.expr() + (carry.expr() * F::from_canonical_u16(1u16 << (8 - shift))),
            ],
            ByteOperation::Rot(a, b, result) => [opcode, a.expr(), b.expr(), result.expr()],
            ByteOperation::RotConst(a, b, c) => [
                opcode,
                a.expr(),
                ArithmeticExpression::from_constant(F::from_canonical_u8(*b)),
                c.expr(),
            ],
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
            ByteOperation::ShrConst(a, _, _) => vec![*a.register()],
            ByteOperation::ShrCarry(a, _, _, _) => vec![*a.register()],
            ByteOperation::Rot(a, b, _) => vec![*a.register(), *b.register()],
            ByteOperation::RotConst(a, _, _) => vec![*a.register()],
            ByteOperation::Not(a, _) => vec![*a.register()],
            ByteOperation::Range(a) => vec![*a.register()],
        }
    }

    pub fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            ByteOperation::And(_, _, c) => vec![*c.register()],
            ByteOperation::Xor(_, _, c) => vec![*c.register()],
            ByteOperation::Shr(_, _, c) => vec![*c.register()],
            ByteOperation::ShrConst(_, _, c) => vec![*c.register()],
            ByteOperation::ShrCarry(_, _, res, c) => vec![*res.register(), *c.register()],
            ByteOperation::Rot(_, _, c) => vec![*c.register()],
            ByteOperation::RotConst(_, _, c) => vec![*c.register()],
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
                let c_val = a_val >> (b_val & 0x7);
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::Shr(a_val, b_val, c_val)
            }
            ByteOperation::ShrConst(a, b, c) => {
                let a_val = from_field(writer.read(a, row_index));
                let c_val = a_val >> (b & 0x7);
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::Shr(a_val, *b, c_val)
            }
            ByteOperation::ShrCarry(a, b, result, carry) => {
                let a_val = from_field(writer.read(a, row_index));
                let b_mod = b & 0x7;
                let (res_val, mut carry_val) = if b_mod != 0 {
                    let res_val = a_val >> b_mod;
                    let carry_val = (a_val << (8 - b_mod)) >> (8 - b_mod);
                    debug_assert_eq!(
                        a_val.rotate_right(b_mod as u32),
                        res_val + (carry_val << (8 - b_mod))
                    );
                    (res_val, carry_val)
                } else {
                    (a_val, 0u8)
                };
                writer.write(result, &as_field(res_val), row_index);
                writer.write(carry, &as_field(carry_val), row_index);

                if carry_val != 0 {
                    carry_val <<= 8 - b_mod;
                }
                ByteOperation::Rot(a_val, *b, res_val + carry_val)
            }
            ByteOperation::Rot(a, b, c) => {
                let a_val = from_field(writer.read(a, row_index));
                let b_val = from_field(writer.read(b, row_index));
                let c_val = a_val.rotate_right((b_val & 0x7) as u32);
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::Rot(a_val, b_val, c_val)
            }
            ByteOperation::RotConst(a, b, c) => {
                let a_val = from_field(writer.read(a, row_index));
                let c_val = a_val.rotate_right((b & 0x7) as u32);
                writer.write(c, &as_field(c_val), row_index);
                ByteOperation::Rot(a_val, *b, c_val)
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
            ByteOperation::ShrConst(_, _, _) => OPCODE_SHR,
            ByteOperation::ShrCarry(_, _, _, _) => OPCODE_ROT,
            ByteOperation::Rot(_, _, _) => OPCODE_ROT,
            ByteOperation::RotConst(_, _, _) => OPCODE_ROT,
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
        let b_val = b & 0x7;
        ByteOperation::Shr(a, b, a >> b_val)
    }

    pub fn rot(a: u8, b: u8) -> Self {
        let b_val = b & 0x7;
        ByteOperation::Rot(a, b, a.rotate_right(b_val as u32))
    }

    pub fn not(a: u8) -> Self {
        ByteOperation::Not(a, !a)
    }

    pub fn range(a: u8) -> Self {
        ByteOperation::Range(a)
    }

    pub fn as_field_op<F: Field>(&self) -> ByteOperation<F> {
        let as_field = |&x| F::from_canonical_u8(x);
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
            ByteOperation::ShrConst(a, b, c) => {
                ByteOperation::ShrConst(as_field(a), *b, as_field(c))
            }
            ByteOperation::ShrCarry(a, b, c, d) => {
                ByteOperation::ShrCarry(as_field(a), *b, as_field(c), as_field(d))
            }
            ByteOperation::Rot(a, b, c) => {
                ByteOperation::Rot(as_field(a), as_field(b), as_field(c))
            }
            ByteOperation::RotConst(a, b, c) => {
                ByteOperation::RotConst(as_field(a), *b, as_field(c))
            }
            ByteOperation::Not(a, b) => ByteOperation::Not(as_field(a), as_field(b)),
            ByteOperation::Range(a) => ByteOperation::Range(as_field(a)),
        }
    }

    pub fn as_field_bits_op<F: Field>(self) -> ByteOperation<[F; 8]> {
        let as_field_bits = |x| u8_to_bits_le(x).map(F::from_canonical_u8);
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
            _ => unreachable!("Const parameters operations cannot convert to field bits"),
        }
    }
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn alloc_public_byte_operation_from_template<T>(
        &mut self,
        op: &ByteOperation<T>,
    ) -> ByteOperation<ByteRegister> {
        match op {
            ByteOperation::And(_, _, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let b = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::And(a, b, result)
            }
            ByteOperation::Xor(_, _, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let b = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::Xor(a, b, result)
            }
            ByteOperation::Shr(_, _, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let b = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::Shr(a, b, result)
            }
            ByteOperation::ShrConst(_, b, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::ShrConst(a, *b, result)
            }
            ByteOperation::ShrCarry(_, b, _, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::ShrCarry(a, *b, result, self.alloc_public::<ByteRegister>())
            }
            ByteOperation::Rot(_, _, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let b = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::Rot(a, b, result)
            }
            ByteOperation::RotConst(_, b, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::RotConst(a, *b, result)
            }
            ByteOperation::Not(_, _) => {
                let a = self.alloc_public::<ByteRegister>();
                let result = self.alloc_public::<ByteRegister>();
                ByteOperation::Not(a, result)
            }
            ByteOperation::Range(_) => {
                let a = self.alloc_public::<ByteRegister>();
                ByteOperation::Range(a)
            }
        }
    }
}
