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
            .bytes()
            .iter()
            .zip(b.bytes().iter())
            .zip(result.bytes().iter())
        {
            let and = ByteOperation::And(a_byte, b_byte, result_byte);
            self.set_byte_operation(&and, operations);
        }
    }

    pub fn set_bitwise_xor<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        b: &ByteArrayRegister<N>,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        for ((a_byte, b_byte), result_byte) in a
            .bytes()
            .iter()
            .zip(b.bytes().iter())
            .zip(result.bytes().iter())
        {
            let xor = ByteOperation::Xor(a_byte, b_byte, result_byte);
            self.set_byte_operation(&xor, operations);
        }
    }

    pub fn set_bitwise_not<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        for (a_byte, result_byte) in a.bytes().iter().zip(result.bytes().iter()) {
            let not = ByteOperation::Not(a_byte, result_byte);
            self.set_byte_operation(&not, operations);
        }
    }

    // pub fn set_bit_shr<const N : usize>(
    //     &mut self,
    //     a: &ByteArrayRegister<N>,
    //     shift: u8,
    //     result: &ByteArrayRegister<N>,
    //     operations: &mut ByteLookupOperations,
    // ) where
    //     L::Instruction: From<ByteOperationInstruction>,
    // {
    //     for (a_byte, result_byte) in a.bytes().iter().zip(result.bytes().iter()) {
    //         let shr = ByteOperation::Shr(a_byte, shift, result_byte);
    //         self.set_byte_operation(&shr, operations);
    //     }
    // }
}
