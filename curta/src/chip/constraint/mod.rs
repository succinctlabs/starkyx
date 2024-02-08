use serde::{Deserialize, Serialize};

use super::arithmetic::ArithmeticConstraint;
use super::instruction::set::AirInstruction;
use super::memory::pointer::accumulate::PointerAccumulator;
use super::register::cubic::CubicRegister;
use super::table::accumulator::Accumulator;
use super::table::bus::channel::BusChannel;
use super::table::bus::global::Bus;
use super::table::lookup::constraint::LookupChipConstraint;
use super::table::powers::Powers;
use super::AirParameters;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::{AirParser, MulParser};
use crate::air::AirConstraint;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint<L: AirParameters> {
    Instruction(AirInstruction<L::Field, L::Instruction>),
    Arithmetic(ArithmeticConstraint<L::Field>),
    Powers(Powers<L::Field, L::CubicParams>),
    Accumulator(Accumulator<L::Field, L::CubicParams>),
    Pointer(PointerAccumulator<L::Field, L::CubicParams>),
    BusChannel(BusChannel<CubicRegister, L::CubicParams>),
    Bus(Bus<CubicRegister, L::CubicParams>),
    Lookup(LookupChipConstraint<L::Field, L::CubicParams>),
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

    pub fn lookup(lookup: LookupChipConstraint<L::Field, L::CubicParams>) -> Self {
        Self::Lookup(lookup)
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
            Constraint::Powers(powers) => powers.eval(parser),
            Constraint::Accumulator(accumulator) => accumulator.eval(parser),
            Constraint::Pointer(accumulator) => accumulator.eval(parser),
            Constraint::BusChannel(bus_channel) => bus_channel.eval(parser),
            Constraint::Bus(bus) => bus.eval(parser),
            Constraint::Lookup(lookup) => lookup.eval(parser),
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

impl<L: AirParameters> From<PointerAccumulator<L::Field, L::CubicParams>> for Constraint<L> {
    fn from(accumulator: PointerAccumulator<L::Field, L::CubicParams>) -> Self {
        Self::Pointer(accumulator)
    }
}

impl<L: AirParameters> From<BusChannel<CubicRegister, L::CubicParams>> for Constraint<L> {
    fn from(bus_channel: BusChannel<CubicRegister, L::CubicParams>) -> Self {
        Self::BusChannel(bus_channel)
    }
}

impl<L: AirParameters> From<Bus<CubicRegister, L::CubicParams>> for Constraint<L> {
    fn from(bus: Bus<CubicRegister, L::CubicParams>) -> Self {
        Self::Bus(bus)
    }
}

impl<L: AirParameters> From<Powers<L::Field, L::CubicParams>> for Constraint<L> {
    fn from(powers: Powers<L::Field, L::CubicParams>) -> Self {
        Self::Powers(powers)
    }
}
