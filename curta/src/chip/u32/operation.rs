// use super::opcode::{U32Opcode, OPCODE_AND, OPCODE_XOR};
// use crate::air::parser::AirParser;
// use crate::air::AirConstraint;
// use crate::chip::builder::AirBuilder;
// use crate::chip::instruction::Instruction;
// use crate::chip::register::array::ArrayRegister;
// use crate::chip::register::bit::BitRegister;
// use crate::chip::register::memory::MemorySlice;
// use crate::chip::trace::writer::TraceWriter;
// use crate::chip::uint::bytes::bit_operations::and::And;
// use crate::chip::uint::bytes::bit_operations::xor::Xor;
// use crate::chip::uint::bytes::bit_operations::BitOperation;
// use crate::chip::AirParameters;
// pub use crate::math::prelude::*;

// #[derive(Debug, Clone)]
// pub enum U32Operation {
//     Bit(BitOperation<32>),
// }

// impl<L: AirParameters> AirBuilder<L> {
//     pub fn u32_operation_from_opcode(&mut self, opcode: &U32Opcode) -> U32Operation
//     where
//         L::Instruction: From<U32Operation>,
//     {
//         let operation = match opcode.ident {
//             OPCODE_AND => {
//                 let a_bits = self.alloc_array::<BitRegister>(32);
//                 let b_bits = self.alloc_array::<BitRegister>(32);
//                 let result_bits = self.alloc_array::<BitRegister>(32);
//                 let and = And {
//                     a: a_bits,
//                     b: b_bits,
//                     result: result_bits,
//                 };
//                 U32Operation::Bit(and.into())
//             }
//             OPCODE_XOR => {
//                 let a_bits = self.alloc_array::<BitRegister>(32);
//                 let b_bits = self.alloc_array::<BitRegister>(32);
//                 let result_bits = self.alloc_array::<BitRegister>(32);
//                 let xor = Xor {
//                     a: a_bits,
//                     b: b_bits,
//                     result: result_bits,
//                 };
//                 U32Operation::Bit(xor.into())
//             }
//             _ => unimplemented!(),
//         };
//         self.register_instruction(operation.clone());
//         operation
//     }
// }

// impl U32Operation {
//     pub fn a_bits(&self) -> ArrayRegister<BitRegister> {
//         match self {
//             U32Operation::Bit(op) => op.a_bits(),
//         }
//     }

//     pub fn b_bits(&self) -> ArrayRegister<BitRegister> {
//         match self {
//             U32Operation::Bit(op) => op.b_bits(),
//         }
//     }

//     pub fn result_bits(&self) -> ArrayRegister<BitRegister> {
//         match self {
//             U32Operation::Bit(op) => op.result_bits(),
//         }
//     }
// }

// impl<AP: AirParser> AirConstraint<AP> for U32Operation {
//     fn eval(&self, parser: &mut AP) {
//         match self {
//             U32Operation::Bit(op) => op.eval(parser),
//         }
//     }
// }

// impl<F: PrimeField64> Instruction<F> for U32Operation {
//     fn inputs(&self) -> Vec<MemorySlice> {
//         match self {
//             U32Operation::Bit(op) => Instruction::<F>::inputs(op),
//         }
//     }

//     fn trace_layout(&self) -> Vec<MemorySlice> {
//         match self {
//             U32Operation::Bit(op) => Instruction::<F>::trace_layout(op),
//         }
//     }

//     fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
//         match self {
//             U32Operation::Bit(op) => Instruction::<F>::write(op, writer, row_index),
//         }
//     }
// }

// impl From<BitOperation<32>> for U32Operation {
//     fn from(op: BitOperation<32>) -> Self {
//         U32Operation::Bit(op)
//     }
// }
