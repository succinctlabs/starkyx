use serde::{Deserialize, Serialize};

use super::writer::{AirWriter, TraceWriter};
use crate::chip::instruction::set::AirInstruction;
use crate::chip::memory::pointer::accumulate::PointerAccumulator;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::table::accumulator::Accumulator;
use crate::chip::table::bus::channel::BusChannel;
use crate::chip::table::bus::global::Bus;
use crate::chip::table::lookup::table::LookupTable;
use crate::chip::table::lookup::values::LookupValues;
use crate::chip::table::powers::Powers;
use crate::chip::AirParameters;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::type_complexity)]
pub struct AirTraceData<L: AirParameters> {
    pub num_challenges: usize,
    pub num_public_inputs: usize,
    pub num_global_values: usize,
    pub execution_trace_length: usize,
    pub instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub global_instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub powers: Vec<Powers<L::Field, L::CubicParams>>,
    pub accumulators: Vec<Accumulator<L::Field, L::CubicParams>>,
    pub pointer_row_accumulators: Vec<PointerAccumulator<L::Field, L::CubicParams>>,
    pub pointer_global_accumulators: Vec<PointerAccumulator<L::Field, L::CubicParams>>,
    pub bus_channels: Vec<BusChannel<CubicRegister, L::CubicParams>>,
    pub buses: Vec<Bus<CubicRegister, L::CubicParams>>,
    pub lookup_values: Vec<LookupValues<L::Field, L::CubicParams>>,
    pub lookup_tables: Vec<LookupTable<L::Field, L::CubicParams>>,
    pub range_data: Option<(
        LookupTable<L::Field, L::CubicParams>,
        LookupValues<L::Field, L::CubicParams>,
    )>,
}

impl<L: AirParameters> AirTraceData<L> {
    #[inline]
    pub fn write_trace_instructions(&self, writer: &mut impl AirWriter<Field = L::Field>) {
        for instruction in self.instructions.iter() {
            writer.write_instruction(instruction);
        }
    }

    #[inline]
    pub fn write_global_instructions(&self, writer: &mut impl AirWriter<Field = L::Field>) {
        for instruction in self.global_instructions.iter() {
            writer.write_instruction(instruction);
        }
    }

    pub fn write_extended_trace(&self, writer: &TraceWriter<L::Field>) {
        let num_rows = writer.read_trace().unwrap().height();

        // Fill in the challenge powers.
        for power in self.powers.iter() {
            writer.write_powers(power);
        }

        // Write accumulations.
        for acc in self.accumulators.iter() {
            writer.write_accumulation(acc);
        }

        // Write pointer accumulations.
        for acc in self.pointer_global_accumulators.iter() {
            writer.write_ptr_accumulation(acc, 0);
        }

        for i in 0..num_rows {
            for acc in self.pointer_row_accumulators.iter() {
                writer.write_ptr_accumulation(acc, i);
            }
        }

        // Write bus channels.
        for channel in self.bus_channels.iter() {
            writer.write_bus_channel(channel);
        }

        // Write bus values.
        for bus in self.buses.iter() {
            writer.write_global_bus(bus);
        }

        // Write lookup tables.
        for table in self.lookup_tables.iter() {
            writer.write_lookup_table(table);
        }

        // Write lookup values.
        for value_data in self.lookup_values.iter() {
            writer.write_lookup_values(value_data);
        }
    }
}
