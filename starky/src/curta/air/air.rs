use super::parser::AirParser;

pub trait Air<P: AirParser> {
    fn eval(self, parser: &P);
}

// pub trait AirConstraint<P: AirParser> {
//     fn assert(&mut self, constraint: &P::Var);
//     fn assert_transition(&mut self, constraint: &P::Var);
// }
