use plonky2::iop::target::Target;

use crate::chip::uint::bytes::operations::value::ByteOperation;

impl ByteOperation<Target> {
    pub fn input_targets(&self) -> Vec<Target> {
        match self {
            ByteOperation::And(a, b, _) => {
                vec![*a, *b]
            }
            ByteOperation::Xor(a, b, _) => {
                vec![*a, *b]
            }
            ByteOperation::Not(a, _) => {
                vec![*a]
            }
            ByteOperation::Shr(a, b, _) => {
                vec![*a, *b]
            }
            ByteOperation::ShrCarry(a, _, _, _) => {
                vec![*a]
            }
            ByteOperation::ShrConst(a, _, _) => {
                vec![*a]
            }
            ByteOperation::Rot(a, b, _) => {
                vec![*a, *b]
            }
            ByteOperation::RotConst(a, _, _) => {
                vec![*a]
            }
            ByteOperation::Range(a) => {
                vec![*a]
            }
        }
    }

    pub fn all_targets(&self) -> Vec<Target> {
        match self {
            ByteOperation::And(a, b, res) => {
                vec![*a, *b, *res]
            }
            ByteOperation::Xor(a, b, res) => {
                vec![*a, *b, *res]
            }
            ByteOperation::Not(a, res) => {
                vec![*a, *res]
            }
            ByteOperation::Shr(a, b, res) => {
                vec![*a, *b, *res]
            }
            ByteOperation::ShrCarry(a, _, res, c) => {
                vec![*a, *res, *c]
            }
            ByteOperation::ShrConst(a, _, res) => {
                vec![*a, *res]
            }
            ByteOperation::Rot(a, b, res) => {
                vec![*a, *b, *res]
            }
            ByteOperation::RotConst(a, _, res) => {
                vec![*a, *res]
            }
            ByteOperation::Range(a) => {
                vec![*a]
            }
        }
    }
}
