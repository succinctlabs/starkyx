use serde::{Deserialize, Serialize};

use crate::chip::register::cubic::CubicRegister;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RawPointerKey<T> {
    pub challenge: CubicRegister,
    pub shift: T,
}

impl<T> RawPointerKey<T> {
    pub(crate) fn new(challenge: CubicRegister, shift: T) -> Self {
        Self { challenge, shift }
    }
}
