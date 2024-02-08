use core::borrow::Borrow;

use super::get::GetInstruction;
use super::instruction::{MemoryInstruction, MemoryOutput, MemorySliceIndex};
use super::pointer::slice::{RawSlice, Slice};
use super::pointer::Pointer;
use super::set::SetInstruction;
use super::time::Time;
use super::value::MemoryValue;
use super::watch::WatchInstruction;
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
        let digest = value.compress(self, ptr.raw, time, &ptr.challenges);
        self.input_to_memory_bus(digest, multiplicity);
        self.unsafe_raw_write(&ptr, *value, multiplicity, true, None);

        ptr
    }

    #[inline]
    pub(crate) fn uninit<V: MemoryValue>(&mut self) -> Pointer<V> {
        let ptr_challenge_powers = self.challenge_powers(3);
        let compression_challenges = self.challenge_powers(V::num_challenges());
        Pointer::from_challenges(ptr_challenge_powers, compression_challenges)
    }

    /// Frees the memory at location `ptr` with value `value` and write time given by `time`.
    pub fn free<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        value: V,
        last_write: &Time<L::Field>,
    ) {
        let digest = value.compress(self, ptr.raw, last_write, &ptr.challenges);
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
            let digest = value.compress(self, ptr.raw, time, &ptr.challenges);
            self.input_to_memory_bus(digest, multiplicity);
            self.unsafe_raw_write(&ptr, *value, multiplicity, true, None);
        }
        slice
    }

    #[inline]
    pub(crate) fn uninit_slice<V: MemoryValue>(&mut self) -> Slice<V> {
        let raw_slice = RawSlice::new(self);
        let compression_challenges = self.challenge_powers(V::num_challenges());
        Slice::new(raw_slice, compression_challenges)
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
    pub fn get<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        last_write_ts: &Time<L::Field>,
        label: Option<String>,
        index: Option<MemorySliceIndex>,
    ) -> V {
        let memory_output = label.map(|label| MemoryOutput {
            label,
            index,
            ts: (*last_write_ts).clone(),
        });

        let value = self.unsafe_raw_read(ptr, memory_output);
        let read_digest = value.compress(self, ptr.raw, last_write_ts, &ptr.challenges);
        self.output_from_memory_bus(read_digest);
        value
    }

    fn unsafe_raw_read<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        memory_output: Option<MemoryOutput<L::Field>>,
    ) -> V {
        let value = self.alloc::<V>();
        let instr = MemoryInstruction::Get(GetInstruction::new(
            ptr.raw,
            *value.register(),
            memory_output,
        ));
        self.register_air_instruction_internal(AirInstruction::mem(instr));
        value
    }

    fn unsafe_raw_write<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        value: V,
        multiplicity: Option<ElementRegister>,
        global: bool,
        memory_output: Option<MemoryOutput<L::Field>>,
    ) {
        let instr = MemoryInstruction::Set(SetInstruction::new(
            ptr.raw,
            *value.register(),
            multiplicity,
            memory_output,
        ));
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
        label: Option<String>,
        index: Option<MemorySliceIndex>,
    ) {
        if value.is_trace() {
            if let Some(mult) = multiplicity {
                assert!(mult.is_trace());
            }
        }
        let write_digest = value.compress(self, ptr.raw, write_ts, &ptr.challenges);
        self.input_to_memory_bus(write_digest, multiplicity);

        let memory_output = label.map(|label| MemoryOutput {
            label,
            index,
            ts: (*write_ts).clone(),
        });

        self.unsafe_raw_write(
            ptr,
            value,
            multiplicity,
            !write_digest.is_trace(),
            memory_output,
        );
    }

    pub fn watch_memory<V: MemoryValue>(&mut self, ptr: &Pointer<V>, name: &str) {
        let instr = MemoryInstruction::Watch(WatchInstruction::new(ptr.raw, name.to_string()));
        self.register_air_instruction_internal(AirInstruction::mem(instr));
    }
}
