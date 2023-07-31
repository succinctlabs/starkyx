use super::Bus;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::Register;
use crate::math::extension::cubic::parameters::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP> for Bus<E> {
    fn eval(&self, parser: &mut AP) {
        let mut acc_prod = parser.one_extension();
        for channel in self.channels.iter() {
            let value = channel.eval(parser);
            acc_prod = parser.mul_extension(acc_prod, value);
        }
        let one = parser.one_extension();
        parser.assert_eq_extension(acc_prod, one);
    }
}
