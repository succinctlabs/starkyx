//! Range checks based on logarithmic derivatives.
//!

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::builder::StarkBuilder;
use super::chip::StarkParameters;
use super::constraint::arithmetic::ArithmeticExpression;
use super::extension::cubic::cubic_expression::CubicExpression;
use super::extension::cubic::register::CubicElementRegister;
use super::extension::cubic::CubicParameters;
use super::register::{ElementRegister, Register, RegisterSerializable};
use super::trace::TraceGenerator;

const BETAS: [u64; 3] = [
    17800306513594245228,
    422882772345461752,
    14491510587541603695,
];

#[derive(Debug, Clone)]
pub struct RangeCheckData {
    table: ElementRegister,
    multiplicity: ElementRegister,
    multiplicity_table_log: CubicElementRegister,
    log_lookup_accumulator: CubicElementRegister,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub(crate) fn arithmetic_range_checks<E: CubicParameters<F>>(&mut self) {
        // assign registers for table, multiplicity and accumulators
        let table = self.alloc::<ElementRegister>();
        self.write_data(&table).unwrap();
        let multiplicity = self.alloc::<ElementRegister>();
        self.write_data(&multiplicity).unwrap();
        let multiplicity_table_log = self.alloc::<CubicElementRegister>();
        self.write_data(&multiplicity_table_log).unwrap();
        let log_lookup_accumulator = self.alloc::<CubicElementRegister>();
        self.write_data(&log_lookup_accumulator).unwrap();

        let one = || -> ArithmeticExpression<F, D> { ArithmeticExpression::one() };
        let zero = || -> ArithmeticExpression<F, D> { ArithmeticExpression::zero() };

        let beta = CubicExpression::from_constants([
            F::from_canonical_u64(BETAS[0]),
            F::from_canonical_u64(BETAS[1]),
            F::from_canonical_u64(BETAS[2]),
        ]);

        // Table constraints
        self.assert_expressions_equal_first_row(table.expr(), zero());
        self.assert_expressions_equal_transition(table.expr() + one(), table.next().expr());

        // Multiplicity inverse constraint
        // multiplicity_table_log = multiplicity / (beta - table))
        self.assert_cubic_expressions_equal(
            CubicExpression::from(multiplicity.expr()),
            multiplicity_table_log.extension_expr() * (beta - table.expr().into()),
        );

        self.range_data = Some(RangeCheckData {
            table,
            multiplicity,
            multiplicity_table_log,
            log_lookup_accumulator,
        });
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceGenerator<F, D> {
    pub fn write_arithmetic_range_checks<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        trace_rows: &mut Vec<Vec<F>>,
        range_data: &RangeCheckData,
    ) -> Result<()> {
        // write table constraints
        for i in 0..num_rows {
            let value = F::from_canonical_usize(i);
            range_data
                .table
                .register()
                .assign(trace_rows, 0, &mut vec![value], i);
        }

        // Calculate multiplicities

        // write multiplicity inverse constraint

        Ok(())
    }
}
