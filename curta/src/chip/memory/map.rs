use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::pointer::RawPointer;
use super::time::TimeStamp;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMap<T>(HashMap<RawPointer, (Vec<T>, TimeStamp<T>)>);

impl<T> MemoryMap<T> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get(&self, ptr: &RawPointer) -> Option<&(Vec<T>, TimeStamp<T>)> {
        self.0.get(ptr)
    }

    pub fn insert(&mut self, ptr: RawPointer, value: Vec<T>, timestamp: TimeStamp<T>) {
        self.0.insert(ptr, (value, timestamp));
    }

    pub fn get_mut(&mut self, ptr: &RawPointer) -> Option<&mut (Vec<T>, TimeStamp<T>)> {
        self.0.get_mut(ptr)
    }
}

impl<T> Default for MemoryMap<T> {
    fn default() -> Self {
        Self::new()
    }
}
