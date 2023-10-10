use super::Bus;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::Register;
use crate::chip::table::log_derivative::constraints::LogConstraints;
use crate::math::extension::cubic::parameters::CubicParameters;

impl<T: EvalCubic, E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for Bus<T, E>
{
    fn eval(&self, parser: &mut AP) {
        let beta = self.challenge.eval(parser);

        // Accumulate the global entries.
        LogConstraints::log_global_accumulation(
            parser,
            beta,
            &self.global_entries,
            &self.global_accumulators,
            self.global_value,
        );

        // Constrain the sum of all entries to be zero.
        let mut sum = self.global_value.eval(parser);
        for value in self.channels.iter() {
            let val = value.eval(parser);
            sum = parser.add_extension(sum, val);
        }
        parser.constraint_extension(sum);
    }
}
