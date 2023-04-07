use alloc::collections::BTreeMap;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use anyhow::{anyhow, Result};

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::util::transpose;

use super::circuit::ChipParameters;
use crate::arithmetic::builder::Chip;
use super::instruction::{Instruction, InstructionID};

use crate::lookup::permuted_cols;

use plonky2_maybe_rayon::*;

#[derive(Debug)]
pub struct TraceHandle<'a, F: RichField + Extendable<D>, const D: usize> {
    instruction_indices: &'a BTreeMap<InstructionID, usize>,
    tx: Sender<(usize, usize, Vec<F>)>,
}

#[derive(Debug)]
pub struct TraceGenerator<F> {
    rx: Receiver<(usize, usize, Vec<F>)>,
}

impl<F: RichField + Extendable<D>, const D: usize> Clone for TraceHandle<'_, F, D> {
    fn clone(&self) -> Self {
        Self {
            instruction_indices: self.instruction_indices,
            tx: self.tx.clone(),
        }
    }
}


impl<F>  TraceGenerator<F> {
    pub fn generate_trace<L, const D: usize>(&self, chip : &Chip<L, F, D>, row_capacity : usize) -> Vec<PolynomialValues<F>> 
    where F : RichField + Extendable<D>,
        L : ChipParameters<F, D> {
           
            // Initiaze the trace with capacity given by the user
            let num_cols = chip.num_columns_no_range_checks();
            let mut trace_rows = vec![vec![F::ZERO; num_cols + 1]; row_capacity];

            while let Ok((i, op_index, mut row)) = self.rx.recv() {
                chip.get(op_index).assign_row(&mut trace_rows, &mut row, i)
            }
    
            // Transpose the trace to get the columns and resize to the correct size
            let mut trace_cols = transpose(&trace_rows);

            // if there are no range checks, return the trace
            if chip.num_range_checks() == 0 {
                return trace_cols.into_par_iter().map(PolynomialValues::new).collect();
            }
            
            // Resize the trace columns to include the range checks
            trace_cols.resize(Chip::<L, F, D>::num_columns(), Vec::with_capacity(row_capacity));

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
                .map(|i| permuted_cols(&trace_values[i], &trace_values[chip.table_index() ]))
                .zip(perm_values.par_iter_mut().chunks(2))
                .for_each(|((col_perm, table_perm), mut trace)| {
                    trace[0].extend(col_perm);
                    trace[1].extend(table_perm);
                });
    
            trace_cols
                .into_par_iter()
                .map(PolynomialValues::new)
                .collect()
        }
}