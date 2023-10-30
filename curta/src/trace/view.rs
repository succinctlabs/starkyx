use super::window::TraceWindow;
use crate::maybe_rayon::*;

#[derive(Debug, Clone)]
pub struct TraceView<'a, T> {
    pub(crate) values: &'a [T],
    pub width: usize,
}

#[derive(Debug)]
pub struct TraceViewMut<'a, T> {
    pub(crate) values: &'a mut [T],
    pub width: usize,
}

impl<'a, T> TraceView<'a, T> {
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
    pub fn window(&'a self, r: usize) -> TraceWindow<'a, T> {
        debug_assert!(r < self.height());
        let last_row = self.height() - 1;
        match r {
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

    pub fn windows(&'a self) -> impl Iterator<Item = TraceWindow<'a, T>> {
        let last_row = self.height() - 1;
        (0..=last_row).map(|r| self.window(r))
    }

    pub fn windows_par_iter(&self) -> impl ParallelIterator<Item = TraceWindow<'_, T>>
    where
        T: Sync,
    {
        let last_row = self.height() - 1;
        (0..=last_row).into_par_iter().map(|r| self.window(r))
    }
}

impl<'a, T> TraceViewMut<'a, T> {
    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }

    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }
}
