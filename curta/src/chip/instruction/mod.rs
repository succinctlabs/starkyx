use super::constraint::Constraint;
use crate::air::parser::AirParser;

pub trait Instruction<AP: AirParser> {
    fn eval(&self, parser: &mut AP) -> Vec<Constraint<AP::Var>>;

    fn constraint_degree() -> usize {
        2
    }
}
