use std::sync::mpsc::Sender;

use super::multiplicity_data::MultiplicityData;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::bytes::operations::instruction::ByteOperationValue;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations<T> {
    tx: Sender<ByteOperationValue<T>>,
    row_acc_challenges: ArrayRegister<CubicRegister>,
    values: Vec<CubicRegister>,
}
