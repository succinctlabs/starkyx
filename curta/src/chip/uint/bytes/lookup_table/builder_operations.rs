use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::register::ByteRegister;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    pub trace_operations: Vec<ByteOperation<ByteRegister>>,
    pub public_operations: Vec<ByteOperation<ByteRegister>>,
}

impl ByteLookupOperations {
    pub fn new() -> Self {
        ByteLookupOperations {
            trace_operations: Vec::new(),
            public_operations: Vec::new(),
        }
    }
}
