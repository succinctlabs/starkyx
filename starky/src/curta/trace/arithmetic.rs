use std::sync::mpsc::Receiver;

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::util::transpose;
use plonky2_maybe_rayon::*;

use super::types::StarkTraceGenerator;
use super::writer::TraceWriter;
use crate::curta::builder::InstructionId;
use crate::curta::chip::{Chip, ChipStark, StarkParameters};
use crate::curta::extension::cubic::{CubicExtension, CubicParameters};
use crate::curta::instruction::Instruction;
use crate::curta::lookup::log_der::LogLookup;
use crate::curta::lookup::Lookup;
use crate::curta::register::RegisterSerializable;

#[derive(Debug)]
pub struct ArithmeticGenerator<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>
{
    trace_rows: Vec<Vec<F>>,
    num_rows: usize,
    rx: Receiver<(usize, InstructionId, Vec<F>)>,
    _marker: core::marker::PhantomData<E>,
}

pub fn trace<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>(
    num_rows: usize,
) -> (TraceWriter<F, D>, ArithmeticGenerator<F, E, D>) {
    let (tx, rx) = std::sync::mpsc::channel();
    (TraceWriter { tx }, ArithmeticGenerator::new(num_rows, rx))
}

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>
    ArithmeticGenerator<F, E, D>
{
    pub fn new(num_rows: usize, rx: Receiver<(usize, InstructionId, Vec<F>)>) -> Self {
        Self {
            trace_rows: Vec::new(),
            num_rows,
            rx,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn range_fn(element: F) -> usize {
        element.to_canonical_u64() as usize
    }

    pub fn generate_trace_rows<L: StarkParameters<F, D>>(
        &self,
        chip: &Chip<L, F, D>,
        row_capacity: usize,
    ) -> Result<Vec<Vec<F>>> {
        // Initiaze the trace with capacity given by the user
        // let num_cols = chip.num_columns_no_range_checks();
        let num_cols = chip.partial_trace_index;
        let mut trace_rows = vec![vec![F::ZERO; num_cols]; row_capacity];

        while let Ok((row_index, id, mut row)) = self.rx.recv() {
            let op_index = chip
                .instruction_indices
                .get(&id)
                .ok_or_else(|| anyhow!("Invalid instruction"))?;
            match id {
                InstructionId::CustomInstruction(_) => {
                    chip.instructions[*op_index].assign_row(&mut trace_rows, &mut row, row_index)
                }
                InstructionId::Write(_) => chip.write_instructions[*op_index].assign_row(
                    &mut trace_rows,
                    &mut row,
                    row_index,
                ),
            };
        }
        Ok(trace_rows)
    }

    pub fn write_lookups(
        num_rows: usize,
        trace_rows: &mut Vec<Vec<F>>,
        beta_slice: &[F],
        lookup_data: &LogLookup,
        table_index: fn(F) -> usize,
    ) -> Result<()> {
        // Get the challenge
        let beta = CubicExtension::<F, E>::from_slice(beta_slice);

        let values_idx = lookup_data.values_idx();

        // Calculate multiplicities
        let mut multiplicities = vec![F::ZERO; num_rows];

        for row in trace_rows.iter() {
            for value in row[values_idx.0..values_idx.1].iter() {
                let index = table_index(*value);
                assert!(index < 1 << 16);
                multiplicities[index] += F::ONE;
            }
        }

        // Write multiplicities into the trace
        let multiplicity = lookup_data.multiplicity.register();
        for i in 0..num_rows {
            multiplicity.assign(trace_rows, 0, &mut [multiplicities[i]], i);
        }

        // Write multiplicity inverse constraint
        let mult_table_log_entries = multiplicities
            .par_iter()
            .enumerate()
            .map(|(i, x)| {
                let table = CubicExtension::from(F::from_canonical_usize(i));
                CubicExtension::from(*x) / (beta - table)
            })
            .collect::<Vec<_>>();

        let mult_table_log = lookup_data.multiplicity_table_log.register();
        for (i, value) in mult_table_log_entries.iter().enumerate() {
            mult_table_log.assign(trace_rows, 0, &value.base_field_array(), i);
        }

        // Log accumulator
        let mut value = CubicExtension::<F, E>::ZERO;
        let split_data = lookup_data.split_data();
        let accumulators = trace_rows
            .par_iter_mut()
            .map(|row| {
                let (values, accumulators) = split_data.split(row);
                let mut accumumulator = CubicExtension::<F, E>::from(F::ZERO);
                for (k, pair) in values.chunks(2).enumerate() {
                    let beta_minus_a = beta - pair[0].into();
                    let beta_minus_b = beta - pair[1].into();
                    accumumulator += beta_minus_a.inverse() + beta_minus_b.inverse();
                    accumulators[3 * k..3 * k + 3]
                        .copy_from_slice(&accumumulator.base_field_array());
                }
                accumumulator
            })
            .collect::<Vec<_>>();

        let log_lookup_next = lookup_data.log_lookup_accumulator.register().next();
        for (i, (acc, mult_table)) in accumulators
            .into_iter()
            .zip(mult_table_log_entries.into_iter())
            .enumerate()
            .filter(|(i, _)| *i != num_rows - 1)
        {
            value += acc - mult_table;
            log_lookup_next.assign(trace_rows, 0, &value.base_field_array(), i);
        }
        Ok(())
    }
}

impl<
        L: StarkParameters<F, D>,
        F: RichField + Extendable<D>,
        E: CubicParameters<F>,
        const D: usize,
    > StarkTraceGenerator<ChipStark<L, F, D>, F, D, 2> for ArithmeticGenerator<F, E, D>
{
    fn generate_round(
        &mut self,
        stark: &ChipStark<L, F, D>,
        round: usize,
        challenges: &[F],
    ) -> Vec<plonky2::field::polynomial::PolynomialValues<F>> {
        match round {
            0 => {
                self.trace_rows = self
                    .generate_trace_rows(&stark.chip, self.num_rows)
                    .unwrap();
                transpose(&self.trace_rows)
                    .into_par_iter()
                    .map(PolynomialValues::new)
                    .collect()
            }
            1 => {
                self.trace_rows
                    .par_iter_mut()
                    .for_each(|row| row.resize(Chip::<L, F, D>::num_columns(), F::ZERO));
                if let Some(table) = stark.chip.range_table {
                    let table_reg = table.register();
                    for i in 0..self.num_rows {
                        let value = F::from_canonical_usize(i);
                        table_reg.assign(&mut self.trace_rows, 0, &[value], i);
                    }
                }

                if let Some(Lookup::LogDerivative(data)) = &stark.chip.range_data {
                    let b_idx = 3 * data.challenge_idx;
                    let beta_slice = &challenges[b_idx..b_idx + 3];
                    Self::write_lookups(
                        self.num_rows,
                        &mut self.trace_rows,
                        beta_slice,
                        data,
                        Self::range_fn,
                    )
                    .unwrap();
                }
                let trace_cols =
                    transpose(&self.trace_rows)[stark.chip.partial_trace_index..].to_vec();
                trace_cols
                    .into_par_iter()
                    .map(PolynomialValues::new)
                    .collect()
            }
            _ => unreachable!("Arithmetic generator only has two rounds"),
        }
    }
}
