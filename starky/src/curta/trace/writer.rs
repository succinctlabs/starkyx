use std::sync::mpsc::Sender;

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::curta::builder::InstructionId;
use crate::curta::instruction::Instruction;
use crate::curta::register::{MemorySlice, Register};

#[derive(Debug)]
pub struct TraceWriter<F, const D: usize>
where
    F: RichField + Extendable<D>,
{
    pub(crate) tx: Sender<(usize, InstructionId, Vec<F>)>,
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

    pub fn write_data<T: Register>(&self, row_index: usize, data: T, row: Vec<F>) -> Result<()> {
        let id = InstructionId::Write(*data.register());
        self.tx
            .send((row_index, id, row))
            .map_err(|_| anyhow!("Failed to send row"))?;
        Ok(())
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
