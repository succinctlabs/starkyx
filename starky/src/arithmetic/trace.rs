use alloc::collections::BTreeMap;
use std::sync::mpsc::{Receiver, Sender};

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::util::transpose;
use plonky2_maybe_rayon::*;

use super::circuit::ChipParameters;
use super::instruction::{Instruction, InstructionID};
use crate::arithmetic::builder::Chip;
use crate::lookup::permuted_cols;

#[derive(Debug)]
pub struct TraceHandle<F> {
    tx: Sender<(usize, InstructionID, Vec<F>)>,
}

#[derive(Debug)]
pub struct TraceGenerator<F: RichField + Extendable<D>, const D: usize> {
    spec: BTreeMap<InstructionID, usize>,
    rx: Receiver<(usize, InstructionID, Vec<F>)>,
}

pub fn trace<F: RichField + Extendable<D>, const D: usize>(
    spec: BTreeMap<InstructionID, usize>,
) -> (TraceHandle<F>, TraceGenerator<F, D>) {
    let (tx, rx) = std::sync::mpsc::channel();
    (TraceHandle { tx }, TraceGenerator { spec, rx })
}

impl<F> Clone for TraceHandle<F> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<F> TraceHandle<F> {
    pub fn write(&self, row_index: usize, instruction: InstructionID, row: Vec<F>) -> Result<()> {
        self.tx
            .send((row_index, instruction, row))
            .map_err(|_| anyhow!("Failed to send row"))?;
        Ok(())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceGenerator<F, D> {
    pub fn generate_trace<L>(
        &self,
        chip: &Chip<L, F, D>,
        row_capacity: usize,
    ) -> Result<Vec<PolynomialValues<F>>>
    where
        F: RichField + Extendable<D>,
        L: ChipParameters<F, D>,
    {
        // Initiaze the trace with capacity given by the user
        let num_cols = chip.num_columns_no_range_checks();
        let mut trace_rows = vec![vec![F::ZERO; num_cols + 1]; row_capacity];

        while let Ok((row_index, id, mut row)) = self.rx.recv() {
            let op_index = self
                .spec
                .get(&id)
                .ok_or_else(|| anyhow!("Invalid instruction"))?;
            chip.get(*op_index)
                .assign_row(&mut trace_rows, &mut row, row_index)
        }

        // Transpose the trace to get the columns and resize to the correct size
        let mut trace_cols = transpose(&trace_rows);

        // if there are no range checks, return the trace
        if chip.num_range_checks() == 0 {
            return Ok(trace_cols
                .into_par_iter()
                .map(PolynomialValues::new)
                .collect());
        }

        // Resize the trace columns to include the range checks
        trace_cols.resize(
            Chip::<L, F, D>::num_columns(),
            Vec::with_capacity(row_capacity),
        );

        // Initialize the table column with the counter 1..=row_capacity
        trace_cols[chip.table_index()]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x = F::from_canonical_usize(i);
            });

        // Calculate the permutation and append permuted columbs to trace
        let (trace_values, perm_values) = trace_cols.split_at_mut(chip.table_index() + 1);
        (0..L::NUM_ARITHMETIC_COLUMNS)
            .into_par_iter()
            .map(|i| permuted_cols(&trace_values[i], &trace_values[chip.table_index()]))
            .zip(perm_values.par_iter_mut().chunks(2))
            .for_each(|((col_perm, table_perm), mut trace)| {
                trace[0].extend(col_perm);
                trace[1].extend(table_perm);
            });

        Ok(trace_cols
            .into_par_iter()
            .map(PolynomialValues::new)
            .collect())
    }
}
