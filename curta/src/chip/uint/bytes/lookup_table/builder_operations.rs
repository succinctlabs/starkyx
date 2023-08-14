use alloc::sync::Arc;
use std::sync::Mutex;

use super::multiplicity_data::MultiplicityData;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    pub multiplicity_data: Arc<Mutex<MultiplicityData>>,
    pub row_acc_challenges: ArrayRegister<CubicRegister>,
    pub values: Vec<CubicRegister>,
}

impl ByteLookupOperations {
    pub fn new(
        multiplicity_data: Arc<Mutex<MultiplicityData>>,
        row_acc_challenges: ArrayRegister<CubicRegister>,
    ) -> Self {
        let values = Vec::new();
        ByteLookupOperations {
            multiplicity_data,
            row_acc_challenges,
            values,
        }
    }
}
