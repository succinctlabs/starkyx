use super::parser::AirParser;

pub trait Curta<AP: AirParser> {
    type Input;
    type Output;

    fn generate_trace(&self, input: &Self::Input) -> Self::Output;
}
