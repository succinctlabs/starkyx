use std::sync::mpsc::Sender;

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::uint::bytes::operations::value::ByteOperation;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations<T> {
    tx: Sender<ByteOperation<T>>,
    row_acc_challenges: ArrayRegister<CubicRegister>,
    values: Vec<CubicRegister>,
}

impl<T> ByteLookupOperations<T> {
    pub fn new(
        tx: Sender<ByteOperation<T>>,
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
