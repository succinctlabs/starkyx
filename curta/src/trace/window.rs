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
            row: 1,
            is_first_row: false,
            is_last_row: false,
        }
    }
}
