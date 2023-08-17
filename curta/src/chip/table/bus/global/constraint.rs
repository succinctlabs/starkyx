use super::Bus;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::Register;
use crate::math::extension::cubic::parameters::CubicParameters;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP> for Bus<E> {
    fn eval(&self, parser: &mut AP) {
        let mut acc_prod = parser.one_extension();

        // Accumulate global inputs
        let beta = self.challenge.eval(parser);
        for register in self.global_inputs.iter() {
            let value = register.eval(parser);
            let beta_minus_value = parser.sub_extension(beta, value);
            acc_prod = parser.mul_extension(acc_prod, beta_minus_value);
        }

        // Accumulate global outputs
        let mut acc_global_outputs = parser.one_extension();
        for register in self.global_outputs.iter() {
            let value = register.eval(parser);
            let beta_minus_value = parser.sub_extension(beta, value);
            acc_global_outputs = parser.mul_extension(acc_global_outputs, beta_minus_value);
        }

        for channel in self.channels.iter() {
            let value = channel.eval(parser);
            acc_prod = parser.mul_extension(acc_prod, value);
        }

        // Final consistency constraint
        parser.assert_eq_extension(acc_prod, acc_global_outputs);
    }
}
