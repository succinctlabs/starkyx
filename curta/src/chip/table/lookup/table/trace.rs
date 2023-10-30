use itertools::Itertools;

use super::{LogLookupTable, LookupTable};
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::cubic::extension::CubicExtension;
use crate::math::prelude::*;
use crate::maybe_rayon::*;

impl<F: PrimeField> TraceWriter<F> {
    /// Writes the table lookups and accumulate assumes multiplicities have been written
    pub(crate) fn write_log_lookup_table<T: EvalCubic, E: CubicParameters<F>>(
        &self,
        table_data: &LogLookupTable<T, F, E>,
    ) -> Vec<CubicExtension<F, E>> {
        let beta = CubicExtension::<F, E>::from(self.read(&table_data.challenge, 0));
        assert_eq!(
            table_data.table.len(),
            table_data.multiplicities_table_log.len()
        );
        assert_eq!(table_data.table.len(), table_data.multiplicities.len());
        let mult_table_log_entries = self
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .map(|row| {
                let mut sum = CubicExtension::ZERO;
                for ((table, multiplicity), table_log_register) in table_data
                    .table
                    .iter()
                    .zip_eq(table_data.multiplicities.iter())
                    .zip_eq(table_data.multiplicities_table_log.iter())
                {
                    let table_val = table.read_from_slice(row);
                    let mult_val = multiplicity.read_from_slice(row);
                    let table = CubicExtension::from(T::trace_value_as_cubic(table_val));
                    let mult = CubicExtension::from(mult_val);
                    let table_log = mult / (beta - table);
                    table_log_register.assign_to_raw_slice(row, &table_log.0);
                    sum += table_log;
                }
                sum
            })
            .collect::<Vec<_>>();

        // Write accumulation
        let mut acc = CubicExtension::ZERO;
        for (i, mult_table) in mult_table_log_entries.iter().enumerate() {
            acc += *mult_table;
            self.write(&table_data.table_accumulator, &acc.0, i);
        }

        // Write the digest value
        self.write(&table_data.digest, &acc.0, self.height - 1);

        mult_table_log_entries
    }

    pub(crate) fn write_lookup_table<E: CubicParameters<F>>(&self, table_data: &LookupTable<F, E>) {
        match table_data {
            LookupTable::Element(table) => {
                self.write_log_lookup_table(table);
            }
            LookupTable::Cubic(table) => {
                self.write_log_lookup_table(table);
            }
        }
    }
}
