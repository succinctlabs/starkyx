use self::arithmetic::ArithmeticConstraint;
use super::instruction::set::AirInstruction;
use super::table::accumulator::Accumulator;
use super::table::bus::channel::BusChannel;
use super::table::bus::global::Bus;
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
    Arithmetic(ArithmeticConstraint<L::Field>),
    Accumulator(Accumulator<L::Field, L::CubicParams>),
    BusChannel(BusChannel<L::Field, L::CubicParams>),
    Bus(Bus<L::CubicParams>),
    Lookup(Box<Lookup<L::Field, L::CubicParams>>),
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

    pub fn lookup(lookup: Lookup<L::Field, L::CubicParams>) -> Self {
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
            // Constraint::Filtered(expression, instruction) => {
            //     assert_eq!(
            //         expression.size, 1,
            //         "Expression multiplying instruction must be of size 1"
            //     );
            //     let element = expression.eval(parser)[0];
            //     let mut mul_parser = MulParser::new(parser, element);
            //     instruction.eval(&mut mul_parser)
            // }
            Constraint::Arithmetic(constraint) => constraint.eval(parser),
            Constraint::Accumulator(accumulator) => accumulator.eval(parser),
            Constraint::BusChannel(bus_channel) => bus_channel.eval(parser),
            Constraint::Bus(bus) => bus.eval(parser),
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

impl<L: AirParameters> From<Accumulator<L::Field, L::CubicParams>> for Constraint<L> {
    fn from(accumulator: Accumulator<L::Field, L::CubicParams>) -> Self {
        Self::Accumulator(accumulator)
    }
}

impl<L: AirParameters> From<BusChannel<L::Field, L::CubicParams>> for Constraint<L> {
    fn from(bus_channel: BusChannel<L::Field, L::CubicParams>) -> Self {
        Self::BusChannel(bus_channel)
    }
}

impl<L: AirParameters> From<Bus<L::CubicParams>> for Constraint<L> {
    fn from(bus: Bus<L::CubicParams>) -> Self {
        Self::Bus(bus)
    }
}
