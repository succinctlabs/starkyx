use self::parser::AirParser;

pub mod parser;
pub mod starky;

pub trait Air<P: AirParser> {
    fn eval(self, parser: &P);
}
