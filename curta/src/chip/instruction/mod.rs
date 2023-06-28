use super::constraint::Constraint;
use crate::air::parser::AirParser;

pub trait Instruction<AP: AirParser> {

    type Constraints: IntoIterator<Item = Constraint<AP::Var>>;

    fn eval(&self, parser: &mut AP) -> Self::Constraints;

    fn constraint_degree() -> usize {
        2
    }
}
