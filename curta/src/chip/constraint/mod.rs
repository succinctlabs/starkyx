use self::arithmetic::expression::ArithmeticExpression;
use self::arithmetic::ArithmeticConstraint;
use super::AirParameters;
use crate::air::parser::{AirParser, MulParser};
use crate::air::AirConstraint;
pub mod arithmetic;

pub enum Constraint<L: AirParameters> {
    Instruction(L::Instruction),
    MulInstruction(ArithmeticExpression<L::Field>, L::Instruction),
    Arithmetic(ArithmeticConstraint<L::Field>),
}

impl<L: AirParameters> Constraint<L> {
    pub fn from_instruction<I>(instruction: I) -> Self
    where
        L::Instruction: From<I>,
    {
        Self::Instruction(instruction.into())
    }
}

impl<L: AirParameters, AP: AirParser<Field = L::Field>> AirConstraint<AP> for Constraint<L>
where
    L::Instruction: AirConstraint<AP> + for<'a> AirConstraint<MulParser<'a, AP>>,
{
    fn eval(&self, parser: &mut AP) {
        match self {
            Constraint::Instruction(instruction) => instruction.eval(parser),
            Constraint::MulInstruction(expression, instruction) => {
                assert!(expression.size == 1);
                let element = expression.eval(parser)[0];
                let mut mul_parser = MulParser::new(parser, element);
                instruction.eval(&mut mul_parser);
            }
            Constraint::Arithmetic(constraint) => constraint.eval(parser),
        }
    }
}
