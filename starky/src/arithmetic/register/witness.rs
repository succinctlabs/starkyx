use super::CellType;

pub struct WitnessData {
    size: usize,
    cell_type: Option<CellType>,
}

impl WitnessData {
    pub fn u16(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: Some(CellType::U16),
        }
    }

    pub fn bitarray(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: Some(CellType::Bit),
        }
    }

    pub fn untyped(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: None,
        }
    }

    pub fn destruct(self) -> (usize, Option<CellType>) {
        (self.size, self.cell_type)
    }
}
