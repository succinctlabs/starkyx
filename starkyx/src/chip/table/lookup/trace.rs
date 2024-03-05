use super::LogLookupTable;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::log_derivative::entry::LogEntry;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::trace::AirTrace;

impl<F: PrimeField> TraceWriter<F> {
    pub fn write_multiplicities_from_fn<E: CubicParameters<F>, T: Register>(
        &self,
        num_rows: usize,
        table_data: &LogLookupTable<T, F, E>,
        table_index: impl Fn(T::Value<F>) -> usize,
        trace_values: &[T],
        public_values: &[T],
    ) {
        // Calculate multiplicities
        let mut multiplicities = vec![F::ZERO; num_rows];

        // Count the multiplicities in the trace
        let trace = self.read_trace().unwrap();
        for row in trace.rows() {
            for value in trace_values.iter() {
                let val = value.read_from_slice(row);
                let index = table_index(val);
                assert!(index < num_rows);
                multiplicities[index] += F::ONE;
            }
        }
        drop(trace);

        // Count the multiplicities in the public values
        let public_slice = self.public.read().unwrap();
        for value in public_values.iter() {
            let val = value.read_from_slice(&public_slice);
            let index = table_index(val);
            assert!(index < num_rows);
            multiplicities[index] += F::ONE;
        }

        // Write multiplicities into the trace
        let multiplicity = table_data.multiplicities.get(0);
        for (i, mult) in multiplicities.iter().enumerate() {
            self.write(&multiplicity, mult, i);
        }
    }

    pub fn get_multiplicities_from_fn<T: EvalCubic>(
        &self,
        num_table_columns: usize,
        num_rows: usize,
        trace_entries: &[LogEntry<T>],
        public_entries: &[LogEntry<T>],
        table_index: impl Fn(T::Value<F>) -> (usize, usize),
    ) -> AirTrace<F> {
        let mut multiplicities_trace = AirTrace::new_with_value(num_table_columns, num_rows, 0u32);

        // Count the multiplicities in the trace
        let trace = self.read_trace().unwrap();
        for row in trace.rows() {
            for entry in trace_entries.iter() {
                let value = entry.value().read_from_slice(row);
                let (row_index, col_index) = table_index(value);
                assert!(col_index < num_table_columns);
                assert!(row_index < num_rows);
                multiplicities_trace.row_mut(row_index)[col_index] += 1;
            }
        }

        // Count the multiplicities in public inputs
        let public_slice = self.public.read().unwrap();
        for entry in public_entries.iter() {
            let value = entry.value().read_from_slice(&public_slice);
            let (row_index, col_index) = table_index(value);
            assert!(col_index < num_table_columns);
            assert!(row_index < num_rows);
            multiplicities_trace.row_mut(row_index)[col_index] += 1;
        }

        AirTrace::from_rows(
            multiplicities_trace
                .values
                .into_par_iter()
                .map(F::from_canonical_u32)
                .collect(),
            num_table_columns,
        )
    }

    pub fn write_lookup_multiplicities<const N: usize>(
        &self,
        multiplicities: ArrayRegister<ElementRegister>,
        values: &[AirTrace<F>; N],
    ) {
        match N {
            0 => panic!("No values to write"),
            1 => {
                let mut trace_write = self.write_trace().unwrap();
                let (start, end) = multiplicities.register().get_range();
                trace_write
                    .rows_par_mut()
                    .zip(values[0].rows_par())
                    .for_each(|(row, values)| {
                        row[start..end].copy_from_slice(values);
                    });
            }
            2 => {
                let mut trace_write = self.write_trace().unwrap();
                let (start, end) = multiplicities.register().get_range();
                trace_write
                    .rows_par_mut()
                    .zip(values[0].rows_par())
                    .zip(values[1].rows_par())
                    .for_each(|((row, values0), values1)| {
                        row[start..end]
                            .iter_mut()
                            .zip(values0)
                            .zip(values1)
                            .for_each(|((row, value0), value1)| {
                                *row = *value0 + *value1;
                            });
                    });
            }
            3 => {
                let mut trace_write = self.write_trace().unwrap();
                let (start, end) = multiplicities.register().get_range();
                trace_write
                    .rows_par_mut()
                    .zip(values[0].rows_par())
                    .zip(values[1].rows_par())
                    .zip(values[2].rows_par())
                    .for_each(|(((row, values0), values1), values2)| {
                        row[start..end]
                            .iter_mut()
                            .zip(values0)
                            .zip(values1)
                            .zip(values2)
                            .for_each(|(((row, value0), value1), value2)| {
                                *row = *value0 + *value1 + *value2;
                            });
                    });
            }
            4 => {
                let mut trace_write = self.write_trace().unwrap();
                let (start, end) = multiplicities.register().get_range();
                trace_write
                    .rows_par_mut()
                    .zip(values[0].rows_par())
                    .zip(values[1].rows_par())
                    .zip(values[2].rows_par())
                    .zip(values[3].rows_par())
                    .for_each(|((((row, values0), values1), values2), values3)| {
                        row[start..end]
                            .iter_mut()
                            .zip(values0)
                            .zip(values1)
                            .zip(values2)
                            .zip(values3)
                            .for_each(|((((row, value0), value1), value2), value3)| {
                                *row = *value0 + *value1 + *value2 + *value3;
                            });
                    });
            }
            5 => {
                let mut trace_write = self.write_trace().unwrap();
                let (start, end) = multiplicities.register().get_range();
                trace_write
                    .rows_par_mut()
                    .zip(values[0].rows_par())
                    .zip(values[1].rows_par())
                    .zip(values[2].rows_par())
                    .zip(values[3].rows_par())
                    .zip(values[4].rows_par())
                    .for_each(
                        |(((((row, values0), values1), values2), values3), values4)| {
                            row[start..end]
                                .iter_mut()
                                .zip(values0)
                                .zip(values1)
                                .zip(values2)
                                .zip(values3)
                                .zip(values4)
                                .for_each(
                                    |(((((row, value0), value1), value2), value3), value4)| {
                                        *row = *value0 + *value1 + *value2 + *value3 + *value4;
                                    },
                                );
                        },
                    );
            }
            6 => {
                let mut trace_write = self.write_trace().unwrap();
                let (start, end) = multiplicities.register().get_range();
                trace_write
                    .rows_par_mut()
                    .zip(values[0].rows_par())
                    .zip(values[1].rows_par())
                    .zip(values[2].rows_par())
                    .zip(values[3].rows_par())
                    .zip(values[4].rows_par())
                    .zip(values[5].rows_par())
                    .for_each(
                        |((((((row, values0), values1), values2), values3), values4), values5)| {
                            row[start..end]
                                .iter_mut()
                                .zip(values0)
                                .zip(values1)
                                .zip(values2)
                                .zip(values3)
                                .zip(values4)
                                .zip(values5)
                                .for_each(
                                    |(
                                        (((((row, value0), value1), value2), value3), value4),
                                        value5,
                                    )| {
                                        *row = *value0
                                            + *value1
                                            + *value2
                                            + *value3
                                            + *value4
                                            + *value5;
                                    },
                                );
                        },
                    );
            }
            _ => unimplemented!("Unsuppored number of values: {}", N),
        }
    }
}
