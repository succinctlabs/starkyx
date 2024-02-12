use core::hash::Hash;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::pointer::key::RawPointerKey;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemEntry<T> {
    pub value: Vec<T>,
    pub multiplicity: T,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMap<T: PartialEq + Eq + Hash>(pub(crate) HashMap<RawPointerKey<T>, MemEntry<T>>);

impl<T: PartialEq + Eq + Hash> MemoryMap<T> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get(&self, ptr: &RawPointerKey<T>) -> Option<&MemEntry<T>> {
        self.0.get(ptr)
    }

    pub fn remove(&mut self, ptr: &RawPointerKey<T>) -> Option<MemEntry<T>> {
        self.0.remove(ptr)
    }

    pub fn insert(&mut self, ptr: RawPointerKey<T>, value: MemEntry<T>) {
        self.0.insert(ptr, value);
    }

    pub fn get_mut(&mut self, ptr: &RawPointerKey<T>) -> Option<&mut MemEntry<T>> {
        self.0.get_mut(ptr)
    }
}

impl<T: PartialEq + Eq + Hash> Default for MemoryMap<T> {
    fn default() -> Self {
        Self::new()
    }
}
