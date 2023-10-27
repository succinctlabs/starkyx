use serde::{Deserialize, Serialize};

use super::params::Ed25519BaseField;
use crate::chip::field::register::FieldRegister;
use crate::chip::register::bit::BitRegister;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedPointRegister {
    pub sign: BitRegister,
    pub y: FieldRegister<Ed25519BaseField>,
}

impl CompressedPointRegister {
    pub fn new(sign: BitRegister, y: FieldRegister<Ed25519BaseField>) -> Self {
        Self { sign, y }
    }
}
