use itertools::Itertools;

use super::Accumulator;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::Register;
use crate::math::extension::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for Accumulator<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let digest = self.digest.eval(parser);

        let values = self
            .values
            .iter()
            .flat_map(|x| x.eval(parser))
            .collect::<Vec<_>>();

        let acc = values.iter().zip_eq(self.challenges.iter()).fold(
            parser.zero_extension(),
            |acc, (val, alpha)| {
                let alpha_val = alpha.eval(parser);
                let val_ext = parser.element_from_base_field(*val);
                let alpha_times_val = parser.mul_extension(alpha_val, val_ext);
                parser.add_extension(acc, alpha_times_val)
            },
        );

        parser.assert_eq_extension(acc, digest);
    }
}
