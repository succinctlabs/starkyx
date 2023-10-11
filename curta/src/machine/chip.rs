use crate::chip::AirParameters;



pub trait Chip {
    type Parameters: AirParameters;
}