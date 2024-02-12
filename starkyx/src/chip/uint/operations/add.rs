use serde::{Deserialize, Serialize};

use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::Register;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::register::{ByteArrayRegister, U32Register, U64Register};
use crate::chip::AirParameters;
use crate::math::prelude::*;

/// Adding byte arrays as elements mod 2^{8 * N}
///
/// Assumes 2^N < FIELD_SIZE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteArrayAdd<const N: usize> {
    pub a: ByteArrayRegister<N>,
    pub b: ByteArrayRegister<N>,
    in_carry: Option<BitRegister>,
    pub result: ByteArrayRegister<N>,
    result_carry: BitRegister,
}

impl<const N: usize> ByteArrayAdd<N> {
    pub fn new(
        a: ByteArrayRegister<N>,
        b: ByteArrayRegister<N>,
        in_carry: Option<BitRegister>,
        result: ByteArrayRegister<N>,
        result_carry: BitRegister,
    ) -> Self {
        Self {
            a,
            b,
            in_carry,
            result,
            result_carry,
        }
    }
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn carrying_add_u32(
        &mut self,
        a: &U32Register,
        b: &U32Register,
        in_carry: &Option<BitRegister>,
        operations: &mut ByteLookupOperations,
    ) -> (U32Register, BitRegister)
    where
        L::Instruction: From<ByteArrayAdd<4>> + From<ByteOperationInstruction>,
    {
        let result = self.alloc::<U32Register>();
        let out_carry = self.alloc::<BitRegister>();
        self.set_add_u32(a, b, in_carry, &result, &out_carry, operations);

        (result, out_carry)
    }

    pub fn add_u32(
        &mut self,
        a: &U32Register,
        b: &U32Register,
        operations: &mut ByteLookupOperations,
    ) -> U32Register
    where
        L::Instruction: From<ByteArrayAdd<4>> + From<ByteOperationInstruction>,
    {
        let (result, _) = self.carrying_add_u32(a, b, &None, operations);
        result
    }

    pub fn set_add_u32(
        &mut self,
        a: &U32Register,
        b: &U32Register,
        in_carry: &Option<BitRegister>,
        result: &U32Register,
        out_carry: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteArrayAdd<4>> + From<ByteOperationInstruction>,
    {
        let add = ByteArrayAdd::<4>::new(*a, *b, *in_carry, *result, *out_carry);
        self.register_instruction(add);

        for byte in result.to_le_bytes() {
            let result_range = ByteOperation::Range(byte);
            self.set_byte_operation(&result_range, operations);
        }
    }

    pub fn set_add_u64(
        &mut self,
        a: &U64Register,
        b: &U64Register,
        in_carry: &Option<BitRegister>,
        result: &U64Register,
        out_carry: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteArrayAdd<4>> + From<ByteOperationInstruction>,
    {
        let result_as_register = result.to_le_limbs::<4>();

        let a_as_register = a.to_le_limbs::<4>();
        let b_as_register = b.to_le_limbs::<4>();

        let lower_carry = self.alloc::<BitRegister>();

        self.set_add_u32(
            &a_as_register.get(0),
            &b_as_register.get(0),
            in_carry,
            &result_as_register.get(0),
            &lower_carry,
            operations,
        );

        self.set_add_u32(
            &a_as_register.get(1),
            &b_as_register.get(1),
            &Some(lower_carry),
            &result_as_register.get(1),
            out_carry,
            operations,
        );
    }

    pub fn carrying_add_u64(
        &mut self,
        a: &U64Register,
        b: &U64Register,
        in_carry: &Option<BitRegister>,
        operations: &mut ByteLookupOperations,
    ) -> (U64Register, BitRegister)
    where
        L::Instruction: From<ByteArrayAdd<4>> + From<ByteOperationInstruction>,
    {
        let result = self.alloc::<U64Register>();
        let out_carry = self.alloc::<BitRegister>();
        self.set_add_u64(a, b, in_carry, &result, &out_carry, operations);

        (result, out_carry)
    }

    pub fn add_u64(
        &mut self,
        a: &U64Register,
        b: &U64Register,
        operations: &mut ByteLookupOperations,
    ) -> U64Register
    where
        L::Instruction: From<ByteArrayAdd<4>> + From<ByteOperationInstruction>,
    {
        let (result, _) = self.carrying_add_u64(a, b, &None, operations);
        result
    }
}

impl<AP: AirParser, const N: usize> AirConstraint<AP> for ByteArrayAdd<N> {
    fn eval(&self, parser: &mut AP) {
        assert!(N <= 4, "ByteArrayAdd<N> only supports N <= 4");
        let a = self.a.eval(parser);
        let b = self.b.eval(parser);
        let in_carry = self.in_carry.map(|x| x.eval(parser));
        let result = self.result.eval(parser);
        let result_carry = self.result_carry.eval(parser);

        let mut a_val = parser.zero();
        let mut b_val = parser.zero();
        let mut result_val = parser.zero();

        for (i, ((a_byte, b_byte), res_byte)) in a.into_iter().zip(b).zip(result).enumerate() {
            let mult = AP::Field::from_canonical_u32(1 << (8 * i));
            let a_byte_times_mult = parser.mul_const(a_byte, mult);
            let b_byte_times_mult = parser.mul_const(b_byte, mult);
            let res_byte_times_mult = parser.mul_const(res_byte, mult);

            a_val = parser.add(a_val, a_byte_times_mult);
            b_val = parser.add(b_val, b_byte_times_mult);
            result_val = parser.add(result_val, res_byte_times_mult);
        }

        let a_plus_b = parser.add(a_val, b_val);
        let a_plus_b_plus_carry = match in_carry {
            Some(carry) => parser.add(a_plus_b, carry),
            None => a_plus_b,
        };
        let two_power = AP::Field::from_canonical_u64(1 << (8 * N));
        let carry_times_mod = parser.mul_const(result_carry, two_power);
        let result_plus_carry = parser.add(result_val, carry_times_mod);
        let constraint = parser.sub(a_plus_b_plus_carry, result_plus_carry);
        parser.constraint(constraint);
    }
}

impl<F: PrimeField64> Instruction<F> for ByteArrayAdd<4> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let a = writer.read(&self.a, row_index);
        let b = writer.read(&self.b, row_index);
        let in_carry = self.in_carry.map(|x| writer.read(&x, row_index));

        let a_val = u32::from_le_bytes(a.map(|x| x.as_canonical_u64() as u8));
        let b_val = u32::from_le_bytes(b.map(|x| x.as_canonical_u64() as u8));
        let in_carry_val = in_carry
            .map(|x| x.as_canonical_u64() as u8 == 1)
            .unwrap_or(false);

        let (result, result_carry) = a_val.carrying_add(b_val, in_carry_val);
        let result_bytes = result.to_le_bytes().map(|x| F::from_canonical_u8(x));

        writer.write(&self.result, &result_bytes, row_index);
        writer.write(
            &self.result_carry,
            &F::from_canonical_u8(result_carry as u8),
            row_index,
        );
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let a = writer.read(&self.a);
        let b = writer.read(&self.b);
        let in_carry = self.in_carry.map(|x| writer.read(&x));

        let a_val = u32::from_le_bytes(a.map(|x| x.as_canonical_u64() as u8));
        let b_val = u32::from_le_bytes(b.map(|x| x.as_canonical_u64() as u8));
        let in_carry_val = in_carry
            .map(|x| x.as_canonical_u64() as u8 == 1)
            .unwrap_or(false);

        let (result, result_carry) = a_val.carrying_add(b_val, in_carry_val);
        let result_bytes = result.to_le_bytes().map(|x| F::from_canonical_u8(x));

        writer.write(&self.result, &result_bytes);
        writer.write(
            &self.result_carry,
            &F::from_canonical_u8(result_carry as u8),
        );
    }
}
