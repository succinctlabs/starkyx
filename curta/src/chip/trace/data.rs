use serde::{Deserialize, Serialize};

use super::writer::TraceWriter;
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::table::accumulator::Accumulator;
use crate::chip::table::bus::channel::BusChannel;
use crate::chip::table::bus::global::Bus;
use crate::chip::table::evaluation::Evaluation;
use crate::chip::table::lookup::table::LookupTable;
use crate::chip::table::lookup::values::LookupValues;
use crate::chip::AirParameters;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::type_complexity)]
pub struct AirTraceData<L: AirParameters> {
    pub num_challenges: usize,
    pub num_public_inputs: usize,
    pub num_global_values: usize,
    pub instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub global_instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub accumulators: Vec<Accumulator<L::Field, L::CubicParams>>,
    pub bus_channels: Vec<BusChannel<CubicRegister, L::CubicParams>>,
    pub buses: Vec<Bus<CubicRegister, L::CubicParams>>,
    pub lookup_values: Vec<LookupValues<L::Field, L::CubicParams>>,
    pub lookup_tables: Vec<LookupTable<L::Field, L::CubicParams>>,
    pub evaluation_data: Vec<Evaluation<L::Field, L::CubicParams>>,
    pub range_data: Option<(
        LookupTable<L::Field, L::CubicParams>,
        LookupValues<L::Field, L::CubicParams>,
    )>,
}

impl<L: AirParameters> AirTraceData<L> {
    pub fn write_extended_trace(&self, _writer: &TraceWriter<L::Field>) {
        todo!()
    }
}
