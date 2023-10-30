use serde::{Deserialize, Serialize};

use super::{LogLookupTable, LogLookupValues};
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum LookupConstraint<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    Table(LogLookupTable<T, F, E>),
    ValuesLocal(LogLookupValues<T, F, E>),
    ValuesGlobal(LogLookupValues<T, F, E>),
    ValuesDigest(CubicRegister, CubicRegister, Option<CubicRegister>),
    Digest(CubicRegister, Vec<CubicRegister>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum LookupChipConstraint<F: Field, E: CubicParameters<F>> {
    Element(LookupConstraint<ElementRegister, F, E>),
    CubicElement(LookupConstraint<CubicRegister, F, E>),
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
            LookupConstraint::Digest(table_digest, element_digests) => {
                let table = table_digest.eval_cubic(parser);
                let elements = element_digests
                    .iter()
                    .map(|b| b.eval_cubic(parser))
                    .collect::<Vec<_>>();
                let mut elem_sum = parser.zero_extension();
                for e in elements {
                    elem_sum = parser.add_extension(elem_sum, e);
                }
                let difference = parser.sub_extension(table, elem_sum);
                parser.constraint_extension_last_row(difference);
            }
        }
    }
}

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LookupChipConstraint<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            LookupChipConstraint::Element(log) => log.eval(parser),
            LookupChipConstraint::CubicElement(log) => log.eval(parser),
        }
    }
}

impl<F: Field, E: CubicParameters<F>> From<LookupConstraint<ElementRegister, F, E>>
    for LookupChipConstraint<F, E>
{
    fn from(constraint: LookupConstraint<ElementRegister, F, E>) -> Self {
        Self::Element(constraint)
    }
}

impl<F: Field, E: CubicParameters<F>> From<LookupConstraint<CubicRegister, F, E>>
    for LookupChipConstraint<F, E>
{
    fn from(constraint: LookupConstraint<CubicRegister, F, E>) -> Self {
        Self::CubicElement(constraint)
    }
}
