#[derive(Debug, Clone)]
pub struct TraceWindow<'a, T> {
    pub local_slice: &'a [T],
    pub next_slice: &'a [T],
    pub row: usize,
    pub is_first_row: bool,
    pub is_last_row: bool,
}
