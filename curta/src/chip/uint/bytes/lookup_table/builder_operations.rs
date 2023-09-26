use alloc::sync::Arc;

use super::multiplicity_data::MultiplicityData;
use crate::chip::register::element::ElementRegister;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    pub multiplicity_data: Arc<MultiplicityData>,
    pub values: Vec<ElementRegister>,
}

impl ByteLookupOperations {
    pub fn new(
        multiplicity_data: Arc<MultiplicityData>,
    ) -> Self {
        let values = Vec::new();
        ByteLookupOperations {
            multiplicity_data,
            values,
        }
    }
}
