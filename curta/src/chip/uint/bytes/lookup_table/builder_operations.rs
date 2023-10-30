use crate::chip::register::element::ElementRegister;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    pub values: Vec<ElementRegister>,
}

impl ByteLookupOperations {
    pub fn new() -> Self {
        ByteLookupOperations { values: Vec::new() }
    }
}
