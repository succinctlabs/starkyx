use serde::{Deserialize, Serialize};

use super::{LogLookupValues, LogLookupTable};
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::Register;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum LookupConstraint<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    Table(LogLookupTable<T, F, E>),
    ValuesLocal(LogLookupValues<T, F, E>),
    ValuesGlobal(LogLookupValues<T, F, E>),
    ValuesDigest(CubicRegister, CubicRegister, Option<CubicRegister>),
    Digest(CubicRegister, CubicRegister),
}

impl<T: EvalCubic, E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LookupConstraint<T, AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            LookupConstraint::Table(table) => table.eval(parser),
            LookupConstraint::ValuesLocal(values) => values.eval(parser),
            LookupConstraint::ValuesGlobal(values) => values.eval_global(parser),
            LookupConstraint::ValuesDigest(digest, local_digest, global_digest) => {
                let digest = digest.eval(parser);
                let local_digest = local_digest.eval(parser);
                let global_digest = global_digest
                    .map(|d| d.eval(parser))
                    .unwrap_or_else(|| parser.zero_extension());

                let mut digest_constraint = parser.add_extension(local_digest, global_digest);
                digest_constraint = parser.sub_extension(digest_constraint, digest);
                parser.constraint_extension_last_row(digest_constraint);
            }
            LookupConstraint::Digest(a, b) => {
                let a = a.eval_cubic(parser);
                let b = b.eval_cubic(parser);
                let difference = parser.sub_extension(a, b);
                parser.constraint_extension_last_row(difference);
            }
        }
    }
}
