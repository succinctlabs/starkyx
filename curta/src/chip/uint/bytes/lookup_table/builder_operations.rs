use std::sync::mpsc::Sender;

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::uint::bytes::operations::value::ByteOperation;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    pub tx: Sender<ByteOperation<u8>>,
    pub row_acc_challenges: ArrayRegister<CubicRegister>,
    pub values: Vec<CubicRegister>,
}

impl ByteLookupOperations {
    pub fn new(
        tx: Sender<ByteOperation<u8>>,
        row_acc_challenges: ArrayRegister<CubicRegister>,
    ) -> Self {
        let values = Vec::new();
        ByteLookupOperations {
            tx,
            row_acc_challenges,
            values,
        }
    }
}
