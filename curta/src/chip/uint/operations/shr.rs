use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::register::ByteArrayRegister;
use crate::chip::AirParameters;

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

        let byte_shift = shift / 8 + 1;
        let zero = ArithmeticExpression::zero();
        for i in N - byte_shift..N {
            self.set_to_expression(&result_bytes.get(i), zero.clone());
        }

        let bit_shift = 8 - (shift % 8);
        for i in 0..N - byte_shift {
            let shr = ByteOperation::ShrConst(
                a_bytes.get(i + byte_shift),
                bit_shift as u8,
                result_bytes.get(i),
            );
            self.set_byte_operation(&shr, operations);
        }
    }
}
