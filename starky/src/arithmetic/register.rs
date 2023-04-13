use super::register2::RegisterType;

pub struct WitnessData {
    size: usize,
    cell_type: Option<RegisterType>,
}

impl WitnessData {
    pub fn u16(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: Some(RegisterType::U16),
        }
    }

    pub fn bitarray(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: Some(RegisterType::Bit),
        }
    }

    pub fn untyped(size: usize) -> Self {
        WitnessData {
            size,
            cell_type: None,
        }
    }

    pub fn destruct(self) -> (usize, Option<RegisterType>) {
        (self.size, self.cell_type)
    }
}
