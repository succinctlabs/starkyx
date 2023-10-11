use crate::chip::{AirParameters, Chip};

pub trait AirChip {
    type Parameters: AirParameters;

    fn air(&self) -> &Chip<Self::Parameters>;
}
