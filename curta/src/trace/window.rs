#[derive(Debug, Clone)]
pub struct TraceWindow<'a, T> {
    pub local_slice: &'a [T],
    pub next_slice: &'a [T],
    pub row: usize,
    pub is_first_row: bool,
    pub is_last_row: bool,
}

impl<'a, T> TraceWindow<'a, T> {
    pub const fn empty() -> Self {
        Self {
            local_slice: &[],
            next_slice: &[],
            row: 0,
            is_first_row: false,
            is_last_row: false,
        }
    }
}

#[derive(Debug)]
pub struct TraceWindowMut<'a, T> {
    pub local_slice: &'a mut [T],
    pub next_slice: &'a mut [T],
    pub row: usize,
    pub is_first_row: bool,
    pub is_last_row: bool,
}

impl<'a, T> TraceWindowMut<'a, T> {
    pub fn empty() -> Self {
        Self {
            local_slice: &mut [],
            next_slice: &mut [],
            row: 0,
            is_first_row: false,
            is_last_row: false,
        }
    }

    pub fn as_window(&self) -> TraceWindow<'_, T> {
        TraceWindow {
            local_slice: self.local_slice,
            next_slice: self.next_slice,
            row: self.row,
            is_first_row: self.is_first_row,
            is_last_row: self.is_last_row,
        }
    }
}

#[derive(Debug)]
pub struct TraceWindowsMutIter<'a, T> {
    values: &'a mut [T],
    width: usize,
    height: usize,
    current_row: usize,
}

impl<'a, T> TraceWindowsMutIter<'a, T> {
    pub(crate) fn new(values: &'a mut [T], width: usize, height: usize) -> Self {
        Self {
            values,
            width,
            height,
            current_row: 0,
        }
    }
}

impl<'a, T> Iterator for TraceWindowsMutIter<'a, T> {
    type Item = TraceWindowMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.height - 1 {
            return None;
        }
        let slice = core::mem::take(&mut self.values);
        let (local_row, rest) = slice.split_at_mut(self.width);
        let (next_row, new_values) = rest.split_at_mut(self.width);
        self.values = new_values;
        let current_row = self.current_row;
        self.current_row += 1;
        Some(TraceWindowMut {
            local_slice: local_row,
            next_slice: next_row,
            row: self.current_row,
            is_first_row: current_row == 0,
            is_last_row: current_row == self.height - 1,
        })
    }
}

impl<'a, T> ExactSizeIterator for TraceWindowsMutIter<'a, T> {
    fn len(&self) -> usize {
        self.height - 1 - self.current_row
    }
}
