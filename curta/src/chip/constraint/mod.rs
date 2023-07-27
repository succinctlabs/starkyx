use self::arithmetic::expression::ArithmeticExpression;
use self::arithmetic::ArithmeticConstraint;
use super::instruction::set::AirInstruction;
use super::table::accumulator::Accumulator;
use super::table::evaluation::Evaluation;
use super::table::lookup::Lookup;
use super::AirParameters;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::{AirParser, MulParser};
use crate::air::AirConstraint;
pub mod arithmetic;

#[derive(Debug, Clone)]
pub enum Constraint<L: AirParameters> {
    Instruction(AirInstruction<L::Field, L::Instruction>),
    MulInstruction(ArithmeticExpression<L::Field>, L::Instruction),
    Arithmetic(ArithmeticConstraint<L::Field>),
    Accumulator(Accumulator<L::CubicParams>),
    Lookup(Box<Lookup<L::Field, L::CubicParams, 1>>),
    Evaluation(Evaluation<L::Field, L::CubicParams>),
}

impl<L: AirParameters> Constraint<L> {
    pub(crate) fn from_instruction_set(
        instruction: AirInstruction<L::Field, L::Instruction>,
    ) -> Self {
        Self::Instruction(instruction)
    }

    pub fn from_instruction<I>(instruction: I) -> Self
    where
        L::Instruction: From<I>,
    {
        Self::Instruction(AirInstruction::CustomInstruction(instruction.into()))
    }

    pub fn lookup(lookup: Lookup<L::Field, L::CubicParams, 1>) -> Self {
        Self::Lookup(Box::new(lookup))
    }

    pub fn evaluation(evalutaion: Evaluation<L::Field, L::CubicParams>) -> Self {
        Self::Evaluation(evalutaion)
    }
}

impl<L: AirParameters, AP: AirParser<Field = L::Field>> AirConstraint<AP> for Constraint<L>
where
    L::Instruction: AirConstraint<AP> + for<'a> AirConstraint<MulParser<'a, AP>>,
    AP: CubicParser<<L as AirParameters>::CubicParams>,
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
            Constraint::Accumulator(accumulator) => accumulator.eval(parser),
            Constraint::Lookup(lookup) => lookup.eval(parser),
            Constraint::Evaluation(evaluation) => evaluation.eval(parser),
        }
    }
}

impl<L: AirParameters> From<ArithmeticConstraint<L::Field>> for Constraint<L> {
    fn from(constraint: ArithmeticConstraint<L::Field>) -> Self {
        Self::Arithmetic(constraint)
    }
}


impl<L: AirParameters> From<Accumulator<L::CubicParams>> for Constraint<L> {
    fn from(accumulator: Accumulator<L::CubicParams>) -> Self {
        Self::Accumulator(accumulator)
    }
}