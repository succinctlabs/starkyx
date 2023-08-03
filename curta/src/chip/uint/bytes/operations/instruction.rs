use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ByteOperationValue<T> {
    And(T, T, T),
    Xor(T, T, T),
    Adc(T, T, T, T, T),
    Shr(T, T, T),
    Shl(T, T, T),
    Not(T, T),
}

impl ByteOperationValue<u8> {
    pub fn and(a: u8, b: u8) -> Self {
        ByteOperationValue::And(a, b, a & b)
    }

    pub fn xor(a: u8, b: u8) -> Self {
        ByteOperationValue::Xor(a, b, a ^ b)
    }

    pub fn adc(a: u8, b: u8, carry: u8) -> Self {
        assert!(carry == 0 || carry == 1);
        let carry_flag = (carry & 1) == 1;
        let (c, res_carry) = a.carrying_add(b, carry_flag);
        ByteOperationValue::Adc(a, b, carry, c, res_carry as u8)
    }

    pub fn shr(a: u8, b: u8) -> Self {
        ByteOperationValue::Shr(a, b, a >> b)
    }

    pub fn shl(a: u8, b: u8) -> Self {
        ByteOperationValue::Shl(a, b, a << b)
    }

    pub fn not(a: u8) -> Self {
        ByteOperationValue::Not(a, !a)
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
            ByteOperationValue::Adc(a, b, c, d, e) => ByteOperationValue::Adc(
                as_field(a),
                as_field(b),
                as_field(c),
                as_field(d),
                as_field(e),
            ),
            ByteOperationValue::Shr(a, b, c) => {
                ByteOperationValue::Shr(as_field(a), as_field(b), as_field(c))
            }
            ByteOperationValue::Shl(a, b, c) => {
                ByteOperationValue::Shl(as_field(a), as_field(b), as_field(c))
            }
            ByteOperationValue::Not(a, b) => ByteOperationValue::Not(as_field(a), as_field(b)),
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
            ByteOperationValue::Adc(a, b, c, d, e) => ByteOperationValue::Adc(
                from_field(a),
                from_field(b),
                from_field(c),
                from_field(d),
                from_field(e),
            ),
            ByteOperationValue::Shr(a, b, c) => {
                ByteOperationValue::Shr(from_field(a), from_field(b), from_field(c))
            }
            ByteOperationValue::Shl(a, b, c) => {
                ByteOperationValue::Shl(from_field(a), from_field(b), from_field(c))
            }
            ByteOperationValue::Not(a, b) => ByteOperationValue::Not(from_field(a), from_field(b)),
        }
    }
}
