pub mod parser;

use parser::AirParser;

pub trait Air<P: AirParser> {
    fn eval(self, parser: &P);
}
