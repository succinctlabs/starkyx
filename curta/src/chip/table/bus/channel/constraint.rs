use super::BusChannel;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::Register;
use crate::chip::table::log_derivative::constraints::LogConstraints;
use crate::math::extension::cubic::parameters::CubicParameters;

impl<T: EvalCubic, AP: CubicParser<E>, E: CubicParameters<AP::Field>> AirConstraint<AP>
    for BusChannel<T, E>
{
    fn eval(&self, parser: &mut AP) {
        let beta = self.challenge.eval(parser);

        // Constrain the trace accumulation of the bus channel.
        LogConstraints::log_trace_accumulation(
            parser,
            beta,
            &self.entries,
            &self.row_accumulators,
            self.table_accumulator,
        );

        // Constrain the out channel to the last row of the bus column.
        let bus_value = self.table_accumulator.eval(parser);
        let out_channel = self.out_channel.eval(parser);
        let out_minus_bus = parser.sub_extension(out_channel, bus_value);
        parser.constraint_extension_last_row(out_minus_bus);
    }
}
