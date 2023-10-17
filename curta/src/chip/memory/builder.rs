use super::get::GetInstruction;
use super::instruction::MemoryInstruction;
use super::pointer::Pointer;
use super::set::SetInstruction;
use super::slice::{Slice, RawSlice};
use super::time::TimeRegister;
use super::value::MemoryValue;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::AirParameters;
use crate::math::prelude::*;

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_time(&mut self, time: TimeRegister) {
        self.time = Some(time)
    }

    pub fn get_at<V: MemoryValue>(&mut self, slice: &Slice<V>, idx: ElementRegister) -> Pointer<V> {
        slice.get_at(self, idx)
    }

    pub fn init_local_memory(&mut self) {
        let mut bus = self.new_bus();
        let channel = bus.new_channel(self);
        assert_eq!(channel, 0);
        assert!(self.buses.is_empty());
        self.buses.push(bus);
    }

    /// Initializes a register with the current time.
    pub fn initialize<V: MemoryValue>(&mut self, value: &V) -> Pointer<V> {
        if value.register().is_trace() {
            panic!("Cannot initialize a trace register");
        }
        let challenge = self.alloc_challenge();
        let ptr = Pointer::from_challenge(challenge);
        let time = self.time.unwrap();
        let digest = value.compress(self, ptr.raw, time);
        self.input_to_memory_bus(digest);

        ptr
    }

    pub fn initialize_slice<V: MemoryValue>(&mut self, values: &[V]) -> Slice<V> {
        let raw_slice = RawSlice::new(self, values.len());
        
        let time = self.time.unwrap();
        for (i, value) in values.iter().enumerate() {
            let digest = value.compress(self, raw_slice.get(i), time);
            self.input_to_memory_bus(digest);
        }
        Slice::new(raw_slice)
    }

    fn input_to_memory_bus(&mut self, digest: CubicRegister) {
        match digest.register() {
            MemorySlice::Local(_, _) => self.input_to_bus(0, digest),
            MemorySlice::Public(_, _) => self.buses[0].insert_global_value(&digest),
            MemorySlice::Global(_, _) => self.buses[0].insert_global_value(&digest),
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

    /// Connects a timestamp `time` from the current row to the timestamp at location `time_next`
    /// in the next row.
    pub fn connect_times(&mut self, time: TimeRegister, time_next: TimeRegister) {
        self.set_to_expression_transition(&time_next.next(), time.expr());
    }

    /// Adancing the timestamp by `amount`. The user is responsible for ensuring that the timestamp
    /// does not overflow in the finite field.
    ///
    /// TODO: a special instruction to update the timestamp.
    pub(crate) fn advance_timestamp(&mut self, amount: usize) {
        let time = self.time.expect("Time register not initialized.");
        let new_time = self.alloc::<TimeRegister>();
        self.set_to_expression(
            &new_time,
            time.expr() + L::Field::from_canonical_usize(amount),
        );
    }

    /// Reads the value from the memory at location `ptr`.
    pub fn get<V: MemoryValue>(&mut self, ptr: &Pointer<V>) -> V {
        let (value, last_write_ts) = self.unsafe_raw_read(ptr);
        let read_digest = value.compress(self, ptr.raw, last_write_ts);
        // TODO: maybe migrate from channel zero to a dedicated channel.
        self.output_from_memory_bus(read_digest);
        // Advance timestamp to mark the event.
        self.advance_timestamp(1);
        let new_write_ts = self.time.unwrap();
        let write_digest = value.compress(self, ptr.raw, new_write_ts);
        self.input_to_memory_bus(write_digest);
        self.unsafe_raw_write(ptr, value, new_write_ts);
        value
    }

    fn unsafe_raw_read<V: MemoryValue>(&mut self, ptr: &Pointer<V>) -> (V, TimeRegister) {
        let value = self.alloc::<V>();
        let time = self.alloc::<TimeRegister>();
        let instr = MemoryInstruction::Get(GetInstruction::new(ptr.raw, *value.register(), time));
        self.register_air_instruction_internal(AirInstruction::mem(instr))
            .unwrap();
        (value, time)
    }

    fn unsafe_raw_write<V: MemoryValue>(&mut self, ptr: &Pointer<V>, value: V, time: TimeRegister) {
        let instr = MemoryInstruction::Set(SetInstruction::new(ptr.raw, *value.register(), time));
        self.register_air_instruction_internal(AirInstruction::mem(instr))
            .unwrap();
    }

    pub fn set<V: MemoryValue>(&mut self, ptr: &Pointer<V>, value: V) {
        let (last_value, last_write_ts) = self.unsafe_raw_read(ptr);
        let read_digest = last_value.compress(self, ptr.raw, last_write_ts);
        self.output_from_memory_bus(read_digest);
        self.advance_timestamp(1);

        let write_ts = self.time.unwrap();
        let write_digest = value.compress(self, ptr.raw, write_ts);
        self.input_to_memory_bus(write_digest)
    }
}
