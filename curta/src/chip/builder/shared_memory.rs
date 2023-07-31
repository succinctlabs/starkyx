use alloc::sync::Arc;
use std::sync::Mutex;

use crate::chip::register::memory::MemorySlice;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SharedMemeoryCore {
    pub global_index: usize,
    pub public_index: usize,
    pub challenge_index: usize,
}

#[derive(Debug, Clone)]
pub struct SharedMemory(Arc<Mutex<SharedMemeoryCore>>);

impl SharedMemory {
    #[inline]
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(SharedMemeoryCore {
            global_index: 0,
            public_index: 0,
            challenge_index: 0,
        })))
    }

    #[inline]
    pub fn global_index(&self) -> usize {
        self.0.lock().unwrap().global_index
    }

    #[inline]
    pub fn challenge_index(&self) -> usize {
        self.0.lock().unwrap().challenge_index
    }

    #[inline]
    pub fn public_index(&self) -> usize {
        self.0.lock().unwrap().public_index
    }

    #[inline]
    pub fn get_global_memory(&self, size: usize) -> MemorySlice {
        let mut core = self.0.lock().unwrap();
        let register = MemorySlice::Global(core.global_index, size);
        core.global_index += size;
        register
    }

    #[inline]
    pub fn get_public_memory(&self, size: usize) -> MemorySlice {
        let mut core = self.0.lock().unwrap();
        let register = MemorySlice::Public(core.public_index, size);
        core.public_index += size;
        register
    }

    #[inline]
    pub fn get_challenge_memory(&self, size: usize) -> MemorySlice {
        let mut core = self.0.lock().unwrap();
        let register = MemorySlice::Challenge(core.challenge_index, size);
        core.challenge_index += size;
        register
    }
}
