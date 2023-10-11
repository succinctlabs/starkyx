
use super::table::ByteLogLookupTable;
use crate::chip::register::element::ElementRegister;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    pub table: ByteLogLookupTable,
    pub values: Vec<ElementRegister>,
}

impl ByteLookupOperations {
    pub fn new(table: ByteLogLookupTable) -> Self {
        ByteLookupOperations {
            table,
            values: Vec::new(),
        }
    }
}
