pub mod commit;

/// A stark trace which is stored as a matrix in row major order
#[derive(Debug, Clone)]
pub struct StarkTrace<T> {
    values: Vec<T>,
    width: usize,
}

#[derive(Debug, Clone)]
pub struct TraceView<'a, T> {
    values: &'a [T],
    width: usize,
}

#[derive(Debug)]
pub struct TraceViewMut<'a, T> {
    values: &'a mut [T],
    width: usize,
}

impl<T> StarkTrace<T> {
    pub fn new(values: Vec<T>, width: usize) -> Self {
        debug_assert_eq!(values.len() % width, 0);
        Self { values, width }
    }

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

    /// Expand the trace, to a minimum of `height` rows.
    pub fn expand_to_height(&mut self, height: usize)
    where
        T: Default + Clone,
    {
        if self.height() < height {
            self.values.resize(self.width * height, T::default());
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.values.chunks_exact(self.width)
    }

    pub fn view(&self) -> TraceView<'_, T> {
        TraceView {
            values: &self.values,
            width: self.width,
        }
    }

    pub fn view_mut(&mut self) -> TraceViewMut<'_, T> {
        TraceViewMut {
            values: &mut self.values,
            width: self.width,
        }
    }
}

impl<'a, T> TraceView<'a, T> {
    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }

    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
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
