use super::BusChannel;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::math::extension::cubic::parameters::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for BusChannel<AP::Field, E>
{
    fn eval(&self, _parser: &mut AP) {
        todo!()
    }
}
