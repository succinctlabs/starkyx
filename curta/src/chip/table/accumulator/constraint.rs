use super::Accumulator;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::plonky2::field::CubicParameters;



impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP> for Accumulator<E> {
    fn eval(&self, parser: &mut AP) {
        todo!()
    }
}


