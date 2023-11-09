pub mod generator;
pub mod view;
pub mod window;
pub mod window_parser;

use core::slice::{ChunksExact, ChunksExactMut};

use plonky2_maybe_rayon::rayon::slice::{
    ChunksExact as ParChunksExact, ChunksExactMut as ParChunksExactMut,
};
use plonky2_maybe_rayon::{
    IndexedParallelIterator, MaybeIntoParIter, MaybeParChunks, MaybeParChunksMut, ParallelIterator,
};
use serde::{Deserialize, Serialize};

use self::view::{TraceView, TraceViewMut};
use self::window::{TraceWindow, TraceWindowMut, TraceWindowsMutIter};

/// A stark trace which is stored as a matrix in row major order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirTrace<T> {
    pub(crate) values: Vec<T>,
    pub(crate) width: usize,
}

impl<T> AirTrace<T> {
    pub const fn new(width: usize) -> Self {
        Self {
            values: Vec::new(),
            width,
        }
    }

    #[inline]
    pub fn new_with_capacity(width: usize, num_rows: usize) -> Self {
        Self {
            values: Vec::with_capacity(width * num_rows),
            width,
        }
    }

    #[inline]
    pub fn new_with_value(width: usize, num_rows: usize, value: T) -> Self
    where
        T: Copy,
    {
        Self {
            values: vec![value; width * num_rows],
            width,
        }
    }

    #[inline]
    pub fn from_rows(values: Vec<T>, width: usize) -> Self {
        debug_assert_eq!(values.len() % width, 0);
        Self { values, width }
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }

    #[inline]
    /// Expand the trace, to a minimum of `height` rows.
    pub fn expand_to_height(&mut self, height: usize)
    where
        T: Default + Clone,
    {
        if self.height() < height {
            self.values.resize(self.width * height, T::default());
        }
    }

    #[inline]
    pub fn rows(&self) -> ChunksExact<'_, T> {
        self.values.chunks_exact(self.width)
    }

    #[inline]
    pub fn rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.values.chunks_exact_mut(self.width)
    }

    #[inline]
    pub fn rows_par(&self) -> ParChunksExact<'_, T>
    where
        T: Send + Sync,
    {
        self.values.par_chunks_exact(self.width)
    }

    #[inline]
    pub fn rows_par_mut(&mut self) -> ParChunksExactMut<'_, T>
    where
        T: Send + Sync,
    {
        self.values.par_chunks_exact_mut(self.width)
    }

    #[inline]
    pub fn view(&self) -> TraceView<'_, T> {
        TraceView {
            values: &self.values,
            width: self.width,
        }
    }

    #[inline]
    pub fn view_mut(&mut self) -> TraceViewMut<'_, T> {
        TraceViewMut {
            values: &mut self.values,
            width: self.width,
        }
    }

    #[inline]
    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = TraceView<'_, T>> {
        let width = self.width;
        self.values
            .chunks_exact(chunk_size * width)
            .map(move |chunk| TraceView {
                values: chunk,
                width,
            })
    }

    #[inline]
    pub fn chunks_mut(&mut self, chunk_size: usize) -> impl Iterator<Item = TraceViewMut<'_, T>> {
        let width = self.width;
        self.values
            .chunks_exact_mut(chunk_size * width)
            .map(move |chunk| TraceViewMut {
                values: chunk,
                width,
            })
    }

    #[inline]
    pub fn chunks_par(&self, chunk_size: usize) -> impl ParallelIterator<Item = TraceView<'_, T>>
    where
        T: Send + Sync,
    {
        let width = self.width;
        self.values
            .par_chunks_exact(chunk_size * width)
            .map(move |chunk| TraceView {
                values: chunk,
                width,
            })
    }

    #[inline]
    pub fn chunks_par_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl IndexedParallelIterator<Item = TraceViewMut<'_, T>> + '_
    where
        T: Send + Sync,
    {
        let width = self.width;
        self.values
            .par_chunks_exact_mut(chunk_size * width)
            .map(move |chunk| TraceViewMut {
                values: chunk,
                width,
            })
    }

    #[inline]
    pub fn window(&self, row: usize) -> TraceWindow<'_, T> {
        debug_assert!(row < self.height());
        let last_row = self.height() - 1;
        match row {
            0 => TraceWindow {
                local_slice: self.row(0),
                next_slice: self.row(1),
                row: 0,
                is_first_row: true,
                is_last_row: last_row == 0,
            },
            r if r == last_row => TraceWindow {
                local_slice: self.row(last_row),
                next_slice: self.row(0),
                row: r,
                is_first_row: false,
                is_last_row: true,
            },
            r => TraceWindow {
                local_slice: self.row(r),
                next_slice: self.row(r + 1),
                row: r,
                is_first_row: false,
                is_last_row: false,
            },
        }
    }

    #[inline]
    pub fn window_mut(&mut self, row: usize) -> TraceWindowMut<'_, T> {
        debug_assert!(row < self.height());
        let last_row = self.height() - 1;
        match row {
            0 => {
                let (first_row, rest) = self.values.split_at_mut(self.width);
                TraceWindowMut {
                    local_slice: first_row,
                    next_slice: &mut rest[..self.width],
                    row: 0,
                    is_first_row: true,
                    is_last_row: last_row == 0,
                }
            }
            r if r == last_row => {
                let (first_row, rest) = self.values.split_at_mut(self.width);
                TraceWindowMut {
                    local_slice: &mut rest[last_row * self.width..],
                    next_slice: first_row,
                    row: r,
                    is_first_row: false,
                    is_last_row: true,
                }
            }
            r => {
                let (first_r_rows, rest) = self.values.split_at_mut((r + 1) * self.width);
                TraceWindowMut {
                    local_slice: &mut first_r_rows[r * self.width..],
                    next_slice: &mut rest[..self.width],
                    row: r,
                    is_first_row: false,
                    is_last_row: false,
                }
            }
        }
    }

    #[inline]
    pub fn windows(&self) -> impl Iterator<Item = TraceWindow<'_, T>> + '_ {
        let last_row = self.height() - 1;
        (0..=last_row).map(|r| self.window(r))
    }

    #[inline]
    pub fn windows_mut(&mut self) -> TraceWindowsMutIter<'_, T> {
        let height = self.height();
        TraceWindowsMutIter::new(&mut self.values, self.width, height)
    }

    #[inline]
    pub fn windows_par_iter(&self) -> impl ParallelIterator<Item = TraceWindow<'_, T>> + '_
    where
        T: Sync,
    {
        let last_row = self.height() - 1;
        (0..=last_row).into_par_iter().map(|r| self.window(r))
    }

    pub fn as_columns(&self) -> Vec<Vec<T>>
    where
        T: Copy,
    {
        if self.width == 0 {
            return Vec::new();
        }
        let mut columns = vec![Vec::with_capacity(self.height()); self.width];
        for row in self.rows() {
            for (i, &v) in row.iter().enumerate() {
                columns[i].push(v);
            }
        }
        columns
    }
}
