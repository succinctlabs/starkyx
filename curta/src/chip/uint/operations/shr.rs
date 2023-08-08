use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::Register;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::uint::register::ByteArrayRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_bit_shr<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        shift: usize,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let a_bytes = a.bytes();
        let result_bytes = result.bytes();

        let shift = shift % (N * 8);
        let byte_shift = shift / 8;
        let bit_shift = shift % 8;

        for i in (N - byte_shift)..N {
            self.assert_zero(&result_bytes.get(i));
        }

        assert!(bit_shift < 8, "Shifts larger than 8 are not supported yet");
        let mult = L::Field::from_canonical_u32(1 << (8 - bit_shift));

        let mut carry = ArithmeticExpression::zero();
        for i in (0..N - byte_shift).rev() {
            let (shift_res, next_carry) =
                (self.alloc::<ByteRegister>(), self.alloc::<ByteRegister>());
            let shr_carry = ByteOperation::ShrCarry(
                a_bytes.get(i + byte_shift),
                bit_shift as u8,
                shift_res,
                next_carry,
            );
            self.set_byte_operation(&shr_carry, operations);
            let expected_res = shift_res.expr() + carry.clone() * mult;
            self.set_to_expression(&result_bytes.get(i), expected_res);
            carry = next_carry.expr();
        }
    }
}
