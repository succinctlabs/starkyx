use alloc::collections::BTreeMap;
use std::sync::mpsc::{Receiver, Sender};

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::util::transpose;
use plonky2_maybe_rayon::*;

use super::builder::InstructionId;
use super::chip::StarkParameters;
use super::instruction::Instruction;
use super::register::{MemorySlice, Register, RegisterSerializable};
use crate::curta::chip::Chip;
use crate::lookup::permuted_cols;

#[derive(Debug)]
pub struct TraceWriter<F, const D: usize>
where
    F: RichField + Extendable<D>,
{
    tx: Sender<(usize, InstructionId, Vec<F>)>,
}

#[derive(Debug)]
pub struct TraceGenerator<F: RichField + Extendable<D>, const D: usize> {
    spec: BTreeMap<InstructionId, usize>,
    rx: Receiver<(usize, InstructionId, Vec<F>)>,
}

pub fn trace<F: RichField + Extendable<D>, const D: usize>(
    spec: BTreeMap<InstructionId, usize>,
) -> (TraceWriter<F, D>, TraceGenerator<F, D>) {
    let (tx, rx) = std::sync::mpsc::channel();
    (TraceWriter { tx }, TraceGenerator { spec, rx })
}

impl<F: RichField + Extendable<D>, const D: usize> Clone for TraceWriter<F, D> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    pub fn write_to_layout<T: Instruction<F, D>>(
        &self,
        row_index: usize,
        instruction: T,
        values: Vec<Vec<F>>,
    ) -> Result<()> {
        let witness_layout = instruction.trace_layout();
        debug_assert!(witness_layout.len() == values.len());
        witness_layout
            .into_iter()
            .zip(values.clone())
            .for_each(|(register, value)| {
                debug_assert!(register.len() == value.len());
            });
        let row = values.into_iter().flatten().collect();
        let id = InstructionId::CustomInstruction(instruction.trace_layout());
        self.tx
            .send((row_index, id, row))
            .map_err(|_| anyhow!("Failed to send row"))?;
        Ok(())
    }

    pub fn write<T: Instruction<F, D>>(
        &self,
        row_index: usize,
        instruction: T,
        row: Vec<F>,
    ) -> Result<()> {
        let id = InstructionId::CustomInstruction(instruction.trace_layout());
        self.tx
            .send((row_index, id, row))
            .map_err(|_| anyhow!("Failed to send row"))?;
        Ok(())
    }

    pub fn write_data<T: RegisterSerializable>(
        &self,
        row_index: usize,
        data: T,
        row: Vec<F>,
    ) -> Result<()> {
        let id = InstructionId::Write(*data.register());
        self.tx
            .send((row_index, id, row))
            .map_err(|_| anyhow!("Failed to send row"))?;
        Ok(())
    }

    pub fn write_data_v2<T: RegisterSerializable>(&self, row_index: usize, data: T, row: Vec<F>) {
        let id = InstructionId::Write(*data.register());
        self.tx.send((row_index, id, row)).unwrap();
    }

    pub fn write_unsafe_raw(
        &self,
        row_index: usize,
        data: &MemorySlice,
        row: Vec<F>,
    ) -> Result<()> {
        let id = InstructionId::Write(*data);
        self.tx
            .send((row_index, id, row))
            .map_err(|_| anyhow!("Failed to send row"))?;
        Ok(())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceGenerator<F, D> {
    pub fn generate_trace<L: StarkParameters<F, D>>(
        &self,
        chip: &Chip<L, F, D>,
        row_capacity: usize,
    ) -> Result<Vec<PolynomialValues<F>>> {
        // Initiaze the trace with capacity given by the user
        let num_cols = chip.num_columns_no_range_checks();
        let mut trace_rows = vec![vec![F::ZERO; num_cols + 1]; row_capacity];

        while let Ok((row_index, id, mut row)) = self.rx.recv() {
            let op_index = self
                .spec
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

        // Transpose the trace to get the columns and resize to the correct size
        let mut trace_cols = transpose(&trace_rows);

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

        // if there are no range checks, return the trace
        if chip.num_range_checks() == 0 {
            return Ok(trace_cols
                .into_par_iter()
                .map(PolynomialValues::new)
                .collect());
        }

        // Calculate the permutation and append permuted columbs to trace
        let (trace_values, perm_values) =
            trace_cols[chip.permutations_range()].split_at_mut(chip.relative_table_index() + 1);
        (0..L::NUM_ARITHMETIC_COLUMNS)
            .into_par_iter()
            .map(|i| permuted_cols(&trace_values[i], &trace_values[chip.relative_table_index()]))
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
