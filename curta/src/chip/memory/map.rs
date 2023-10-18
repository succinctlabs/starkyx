use core::hash::Hash;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::pointer::key::RawPointerKey;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMap<T: PartialEq + Eq + Hash>(pub(crate) HashMap<RawPointerKey<T>, Vec<T>>);

impl<T: PartialEq + Eq + Hash> MemoryMap<T> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get(&self, ptr: &RawPointerKey<T>) -> Option<&Vec<T>> {
        self.0.get(ptr)
    }

    pub fn insert(&mut self, ptr: RawPointerKey<T>, value: Vec<T>) {
        self.0.insert(ptr, value);
    }

    pub fn get_mut(&mut self, ptr: &RawPointerKey<T>) -> Option<&mut Vec<T>> {
        self.0.get_mut(ptr)
    }
}

impl<T: PartialEq + Eq + Hash> Default for MemoryMap<T> {
    fn default() -> Self {
        Self::new()
    }
}
