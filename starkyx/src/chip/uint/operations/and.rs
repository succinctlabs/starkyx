use crate::chip::builder::AirBuilder;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::register::ByteArrayRegister;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_bitwise_and<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        b: &ByteArrayRegister<N>,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        for ((a_byte, b_byte), result_byte) in a
            .to_le_bytes()
            .iter()
            .zip(b.to_le_bytes().iter())
            .zip(result.to_le_bytes().iter())
        {
            let and = ByteOperation::And(a_byte, b_byte, result_byte);
            self.set_byte_operation(&and, operations);
        }
    }

    pub fn bitwise_and<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        b: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) -> ByteArrayRegister<N>
    where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let result = self.alloc::<ByteArrayRegister<N>>();
        self.set_bitwise_and(a, b, &result, operations);
        result
    }
}
