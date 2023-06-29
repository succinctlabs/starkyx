use crate::air::parser::AirParser;
use crate::challenger::Challenger;

pub trait StarkConfig<AP: AirParser> {
    type Challenger: Challenger<AP>;

    type Proof;
}
