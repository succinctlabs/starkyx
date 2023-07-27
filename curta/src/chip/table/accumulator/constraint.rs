use super::Accumulator;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::plonky2::field::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP> for Accumulator<E> {
    fn eval(&self, parser: &mut AP) {
        let digest = self.digest.eval(parser);

        let values_array = self
            .values
            .iter()
            .map(|x| ArrayRegister::<ElementRegister>::from_register_unsafe(*x))
            .flatten();

        let acc = values_array.zip(self.challenges).fold(
            parser.zero_extension(),
            |acc, (value, alpha)| {
                let value = value.eval(parser);
                let alpha = alpha.eval(parser);
                let value = parser.element_from_base_field(value);
                let alpha_times_value = parser.mul_extension(value, alpha);
                parser.add_extension(acc, alpha_times_value)
            },
        );

        parser.assert_eq_extension(acc, digest);
    }
}
