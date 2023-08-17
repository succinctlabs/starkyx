use core::array::from_fn;

use crate::chip::builder::AirBuilder;
use crate::chip::register::Register;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::uint::register::ByteArrayRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_bit_rotate_right<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        rotation: usize,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let result_bytes = result.to_le_bytes();

        let rotation = rotation % (N * 8);
        let byte_rotation = rotation / 8;
        let bit_rotation = rotation % 8;

        let mult = L::Field::from_canonical_u32(1 << (8 - bit_rotation));

        let a_bytes = a.to_le_bytes();
        let a_bytes_rotated: [_; N] = from_fn(|i| a_bytes.get((i + byte_rotation) % N));

        let (last_rot, last_carry) = (self.alloc::<ByteRegister>(), self.alloc::<ByteRegister>());
        let shr_carry = ByteOperation::ShrCarry(
            a_bytes_rotated[N - 1],
            bit_rotation as u8,
            last_rot,
            last_carry,
        );
        self.set_byte_operation(&shr_carry, operations);

        let mut carry = last_carry.expr();
        for i in (0..N - 1).rev() {
            let (shift_res, next_carry) =
                (self.alloc::<ByteRegister>(), self.alloc::<ByteRegister>());
            let shr_carry = ByteOperation::ShrCarry(
                a_bytes_rotated[i],
                bit_rotation as u8,
                shift_res,
                next_carry,
            );
            self.set_byte_operation(&shr_carry, operations);
            let expected_res = shift_res.expr() + carry.clone() * mult;
            self.set_to_expression(&result_bytes.get(i), expected_res);
            carry = next_carry.expr();
        }

        // Constraint the last byte with the carry from the first
        let expected_res = last_rot.expr() + carry.clone() * mult;
        self.set_to_expression(&result_bytes.get(N - 1), expected_res);
    }

    pub fn bit_rotate_right<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        rotation: usize,
        operations: &mut ByteLookupOperations,
    ) -> ByteArrayRegister<N>
    where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let result = self.alloc::<ByteArrayRegister<N>>();
        self.set_bit_rotate_right(a, rotation, &result, operations);
        result
    }
}
