use core::hash::Hash;

use plonky2_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator};

use super::public::PublicWriter;
use super::row::RowWriter;
use super::window::WindowWriter;
use crate::chip::memory::map::MemoryMap;
use crate::chip::trace::data::AirTraceData;
use crate::chip::AirParameters;
use crate::math::field::Field;
use crate::trace::view::TraceViewMut;
use crate::trace::AirTrace;

#[derive(Debug, Clone)]
pub struct AirWriterData<T: PartialEq + Eq + Hash> {
    pub trace: AirTrace<T>,
    pub public: Vec<T>,
    pub(crate) memory: MemoryMap<T>,
}

#[derive(Debug)]
pub struct AirWriterChunkMut<'a, T: PartialEq + Eq + Hash> {
    pub trace: TraceViewMut<'a, T>,
    pub public: &'a [T],
    pub(crate) memory: MemoryMap<T>,
    pub height: usize,
    pub initial_row: usize,
}

impl<T: PartialEq + Eq + Hash> AirWriterData<T> {
    #[inline]
    pub fn new<L: AirParameters<Field = T>>(air_data: &AirTraceData<L>, num_rows: usize) -> Self
    where
        T: Field,
    {
        let num_public_inputs = air_data.num_public_inputs;
        Self::new_with_value(
            L::Field::ZERO,
            air_data.execution_trace_length,
            num_rows,
            num_public_inputs,
        )
    }

    #[inline]
    pub fn new_with_value(value: T, width: usize, num_rows: usize, num_public_inputs: usize) -> Self
    where
        T: Copy,
    {
        Self {
            trace: AirTrace::new_with_value(width, num_rows, value),
            public: vec![value; num_public_inputs],
            memory: MemoryMap::new(),
        }
    }

    #[inline]
    pub fn public_writer(&mut self) -> PublicWriter<'_, T> {
        PublicWriter::new(&mut self.public, &mut self.memory, self.trace.height())
    }

    #[inline]
    pub fn chunks(
        &mut self,
        chunk_size: usize,
    ) -> impl Iterator<Item = AirWriterChunkMut<'_, T>> + '_
    where
        T: Clone + Send + Sync,
    {
        let height = self.trace.height();
        assert_eq!(height % chunk_size, 0);
        let num_chunks = height / chunk_size;
        self.trace
            .chunks_mut(chunk_size)
            .zip((0..num_chunks).map(move |i| (i, chunk_size, height)))
            .map(|(chunk, (i, size, height))| AirWriterChunkMut {
                trace: chunk,
                public: &self.public,
                memory: self.memory.clone(),
                height,
                initial_row: i * size,
            })
    }

    #[inline]
    pub fn chunks_par(
        &mut self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = AirWriterChunkMut<'_, T>> + '_
    where
        T: Clone + Send + Sync,
    {
        let height = self.trace.height();
        assert_eq!(height % chunk_size, 0);
        let num_chunks = height / chunk_size;
        self.trace
            .chunks_par_mut(chunk_size)
            .zip_eq(
                (0..num_chunks)
                    .into_par_iter()
                    .map(move |i| (i, chunk_size, height)),
            )
            .map(|(chunk, (i, size, height))| AirWriterChunkMut {
                trace: chunk,
                public: &self.public,
                memory: self.memory.clone(),
                height,
                initial_row: i * size,
            })
    }
}

impl<'a, T: PartialEq + Eq + Hash> AirWriterChunkMut<'a, T> {
    #[inline]
    pub fn row_writer(&mut self, row_index: usize) -> RowWriter<'_, T> {
        RowWriter::new(
            self.trace.row_mut(row_index),
            self.public,
            &mut self.memory,
            row_index + self.initial_row,
            self.height,
        )
    }

    #[inline]
    pub fn window_writer(&mut self, row_index: usize) -> WindowWriter<'_, T> {
        WindowWriter::new(
            self.trace.window_mut(row_index),
            self.public,
            &mut self.memory,
            row_index + self.initial_row,
            self.height,
        )
    }
}
