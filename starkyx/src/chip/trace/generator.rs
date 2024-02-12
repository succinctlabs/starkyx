use alloc::sync::Arc;

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};

use super::data::AirTraceData;
use super::writer::TraceWriter;
use crate::chip::table::log_derivative::entry::LogEntry;
use crate::chip::table::lookup::table::LookupTable;
use crate::chip::table::lookup::values::LookupValues;
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::trace::generator::TraceGenerator;
use crate::trace::AirTrace;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ArithmeticGenerator<L: AirParameters> {
    pub writer: TraceWriter<L::Field>,
    pub air_data: AirTraceData<L>,
    pub num_rows: usize,
}

impl<L: AirParameters> ArithmeticGenerator<L> {
    pub fn reset(&self) {
        let trace_new = AirTrace::new_with_capacity(L::num_columns(), self.num_rows);
        let global_new = Vec::new();
        let challenges_new = Vec::new();
        let public_new = Vec::new();

        let mut trace = self.writer.0.trace.write().unwrap();
        *trace = trace_new;

        let mut global = self.writer.0.global.write().unwrap();
        *global = global_new;

        let mut challenges = self.writer.0.challenges.write().unwrap();
        *challenges = challenges_new;

        let mut public = self.writer.0.public.write().unwrap();
        *public = public_new;
    }

    pub fn new(air_data: AirTraceData<L>, num_rows: usize) -> Self {
        let num_public_inputs = air_data.num_public_inputs;
        let num_global_values = air_data.num_global_values;
        Self {
            writer: TraceWriter::new_with_value(
                L::Field::ZERO,
                L::num_columns(),
                num_rows,
                num_public_inputs,
                num_global_values,
            ),
            air_data,
            num_rows,
        }
    }

    #[inline]
    pub fn range_fn(element: L::Field) -> usize {
        element.as_canonical_u64() as usize
    }

    pub fn new_writer(&self) -> TraceWriter<L::Field> {
        self.writer.clone()
    }

    pub fn trace_clone(&self) -> AirTrace<L::Field> {
        self.writer.read_trace().unwrap().clone()
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.writer.0)
    }
}

impl<L: AirParameters> TraceGenerator<L::Field, Chip<L>> for ArithmeticGenerator<L> {
    type Error = Error;

    fn generate_round(
        &self,
        air: &Chip<L>,
        round: usize,
        challenges: &[L::Field],
        global_values: &mut [L::Field],
        public_inputs: &[L::Field],
    ) -> Result<AirTrace<L::Field>> {
        match round {
            0 => {
                let (id_0, id_1) = (0, self.air_data.num_public_inputs);
                let mut public_write = self.writer.0.public.write().unwrap();
                public_write[id_0..id_1].copy_from_slice(&public_inputs[id_0..id_1]);
                drop(public_write);

                let num_rows = self.num_rows;

                // Write the range check table and multiplicitiies
                if let Some((LookupTable::Element(table), LookupValues::Element(values))) =
                    &self.air_data.range_data
                {
                    assert_eq!(table.table.len(), 1);
                    let table_column = table.table[0];
                    for i in 0..num_rows {
                        self.writer
                            .write(&table_column, &L::Field::from_canonical_usize(i), i);
                    }

                    self.writer.write_multiplicities_from_fn(
                        num_rows,
                        table,
                        Self::range_fn,
                        &values
                            .trace_values
                            .iter()
                            .map(LogEntry::value)
                            .copied()
                            .collect::<Vec<_>>(),
                        &values
                            .public_values
                            .iter()
                            .map(LogEntry::value)
                            .copied()
                            .collect::<Vec<_>>(),
                    );
                }

                let trace = self.trace_clone();
                let execution_trace_values = trace
                    .rows_par()
                    .flat_map(|row| row[..air.execution_trace_length].to_vec())
                    .collect::<Vec<_>>();
                Ok(AirTrace {
                    values: execution_trace_values,
                    width: air.execution_trace_length,
                })
            }
            1 => {
                // Insert the challenges into the generator
                let writer = self.new_writer();
                let mut challenges_write = writer.challenges.write().unwrap();
                challenges_write.extend_from_slice(challenges);
                drop(challenges_write);

                self.air_data.write_extended_trace(&writer);

                let trace = self.trace_clone();
                let extended_trace_values = trace
                    .rows_par()
                    .flat_map(|row| row[air.execution_trace_length..].to_vec())
                    .collect::<Vec<_>>();

                let new_global = self.writer.0.global.read().unwrap();
                let (id_0, id_1) = (0, air.num_global_values);
                global_values[id_0..id_1].copy_from_slice(&new_global[id_0..id_1]);
                drop(new_global);
                Ok(AirTrace {
                    values: extended_trace_values,
                    width: L::num_columns() - air.execution_trace_length,
                })
            }
            _ => unreachable!("Chip air IOP only has two rounds"),
        }
    }
}
