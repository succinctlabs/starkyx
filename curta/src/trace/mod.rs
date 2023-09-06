pub mod generator;
pub mod view;
pub mod window;
pub mod window_parser;

use core::slice::{ChunksExact, ChunksExactMut};

use plonky2_maybe_rayon::rayon::slice::{
    ChunksExact as ParChunksExact, ChunksExactMut as ParChunksExactMut,
};
use serde::{Deserialize, Serialize};

use self::view::{TraceView, TraceViewMut};
use self::window::TraceWindow;
use crate::maybe_rayon::*;

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
    pub fn windows_iter(&self) -> impl Iterator<Item = TraceWindow<'_, T>> {
        let last_row = self.height() - 1;
        (0..=last_row).map(|r| self.window(r))
    }

    #[inline]
    pub fn windows_par_iter(&self) -> impl ParallelIterator<Item = TraceWindow<'_, T>>
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
