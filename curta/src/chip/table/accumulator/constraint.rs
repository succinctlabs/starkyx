use super::Accumulator;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::plonky2::field::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP> for Accumulator<E> {
    fn eval(&self, parser: &mut AP) {
        let digest = self.digest.eval_extension(parser);

        let values_array = self
            .values
            .iter()
            .map(|x| ArrayRegister::<ElementRegister>::from_register_unsafe(*x))
            .flat_map(|x| x.into_iter());

        let mut acc = parser.zero_extension();
        for (value, alpha) in values_array.zip(self.challenges) {
            let value = value.eval(parser);
            let alpha = alpha.eval_extension(parser);
            let value = parser.element_from_base_field(value);
            let alpha_times_value = parser.mul_extension(value, alpha);
            acc = parser.add_extension(acc, alpha_times_value);
        }

        parser.assert_eq_extension(acc, digest);
    }
}
