use core::borrow::Borrow;

use super::get::GetInstruction;
use super::instruction::MemoryInstruction;
use super::pointer::slice::{RawSlice, Slice};
use super::pointer::Pointer;
use super::set::SetInstruction;
use super::time::Time;
use super::value::MemoryValue;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::slice::RegisterSlice;
use crate::chip::register::RegisterSerializable;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    /// Initializes the memory bus.
    pub fn init_local_memory(&mut self) {
        let mut bus = self.new_bus();
        let channel = bus.new_channel(self);
        assert_eq!(channel, 0);
        assert!(self.buses.is_empty());
        self.buses.push(bus);
    }

    /// Initializes a pointer with initial `value` and write time given by `time`.
    pub fn initialize<V: MemoryValue>(
        &mut self,
        value: &V,
        time: &Time<L::Field>,
        multiplicity: Option<ElementRegister>,
    ) -> Pointer<V> {
        if value.register().is_trace() {
            panic!("Cannot initialize a trace register");
        }
        let ptr = self.uninit();
        let digest = value.compress(self, ptr.raw, time);
        self.input_to_memory_bus(digest, multiplicity);
        self.unsafe_raw_write(&ptr, *value, true);

        ptr
    }

    #[inline]
    pub(crate) fn uninit<V: MemoryValue>(&mut self) -> Pointer<V> {
        let challenge = self.alloc_challenge();
        Pointer::from_challenge(challenge)
    }

    /// Frees the memory at location `ptr` with value `value` and write time given by `time`.
    pub fn free<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        value: V,
        last_write: &Time<L::Field>,
    ) {
        let digest = value.compress(self, ptr.raw, last_write);
        self.output_from_memory_bus(digest)
    }

    /// Initializes a slice with initial `values` and write time given by `time`.
    pub fn initialize_slice<V: MemoryValue>(
        &mut self,
        values: &impl RegisterSlice<V>,
        time: &Time<L::Field>,
        multiplicity: Option<ElementRegister>,
    ) -> Slice<V> {
        let slice = self.uninit_slice();

        for (i, value) in values.value_iter().enumerate() {
            let value = value.borrow();
            let ptr = slice.get(i);
            let digest = value.compress(self, ptr.raw, time);
            self.input_to_memory_bus(digest, multiplicity);
            self.unsafe_raw_write(&ptr, *value, true);
        }
        slice
    }

    #[inline]
    pub(crate) fn uninit_slice<V: MemoryValue>(&mut self) -> Slice<V> {
        let raw_slice = RawSlice::new(self);
        Slice::new(raw_slice)
    }

    fn input_to_memory_bus(
        &mut self,
        digest: CubicRegister,
        multiplicity: Option<ElementRegister>,
    ) {
        match (digest.register(), multiplicity) {
            (MemorySlice::Local(_, _), None) => self.input_to_bus(0, digest),
            (MemorySlice::Public(_, _), None) => self.buses[0].insert_global_value(&digest),
            (MemorySlice::Global(_, _), None) => self.buses[0].insert_global_value(&digest),
            (MemorySlice::Local(_, _), Some(m)) => {
                self.input_to_bus_with_multiplicity(0, digest, m)
            }
            (MemorySlice::Public(_, _), Some(m)) => {
                self.buses[0].insert_global_value_with_multiplicity(&digest, m)
            }
            (MemorySlice::Global(_, _), Some(m)) => {
                self.buses[0].insert_global_value_with_multiplicity(&digest, m)
            }
            _ => panic!("Expected local, public, or global register"),
        }
    }

    fn output_from_memory_bus(&mut self, digest: CubicRegister) {
        match digest.register() {
            MemorySlice::Local(_, _) => self.output_from_bus(0, digest),
            MemorySlice::Public(_, _) => self.buses[0].output_global_value(&digest),
            MemorySlice::Global(_, _) => self.buses[0].output_global_value(&digest),
            _ => panic!("Expected local, public, or global register"),
        }
    }

    /// Reads the value from the memory at location `ptr`.
    pub fn get<V: MemoryValue>(&mut self, ptr: &Pointer<V>, last_write_ts: &Time<L::Field>) -> V {
        let value = self.unsafe_raw_read(ptr);
        let read_digest = value.compress(self, ptr.raw, last_write_ts);
        self.output_from_memory_bus(read_digest);
        value
    }

    fn unsafe_raw_read<V: MemoryValue>(&mut self, ptr: &Pointer<V>) -> V {
        let value = self.alloc::<V>();
        let instr = MemoryInstruction::Get(GetInstruction::new(ptr.raw, *value.register()));
        self.register_air_instruction_internal(AirInstruction::mem(instr));
        value
    }

    fn unsafe_raw_write<V: MemoryValue>(&mut self, ptr: &Pointer<V>, value: V, global: bool) {
        let instr = MemoryInstruction::Set(SetInstruction::new(ptr.raw, *value.register()));
        if global {
            self.register_global_air_instruction_internal(AirInstruction::mem(instr))
        } else {
            self.register_air_instruction_internal(AirInstruction::mem(instr))
        }
    }

    /// Writes `value` to the memory at location `ptr` with write time given by `write_ts`. Values
    /// can be written with an optional `multiplicity`.
    ///
    /// If `multiplicity` is `None`, then the value is written to the memory bus with multiplicity
    /// set to 1 allowing a single read. If `multiplicity` is `Some(m)`, then the value is written
    /// to the memory bus with multiplicity given by the value of `m`, allowing `m` reads.
    pub fn set<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        value: V,
        write_ts: &Time<L::Field>,
        multiplicity: Option<ElementRegister>,
    ) {
        if value.is_trace() {
            if let Some(mult) = multiplicity {
                assert!(mult.is_trace());
            }
        }
        let write_digest = value.compress(self, ptr.raw, write_ts);
        self.input_to_memory_bus(write_digest, multiplicity);
        self.unsafe_raw_write(ptr, value, !write_digest.is_trace())
    }
}
