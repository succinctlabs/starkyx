use crate::air::parser::AirParser;
use crate::math::prelude::*;

pub fn byte_decomposition<AP: AirParser, const N: usize>(
    element: AP::Var,
    values: &[AP::Var; N],
    parser: &mut AP,
) -> AP::Var {
    let two_powers: [_; N] = core::array::from_fn(|i| AP::Field::from_canonical_u32(1 << (8 * i)));

    let mut element_value = parser.zero();
    for (value, power) in values.iter().zip(two_powers) {
        let value_two_pow = parser.mul_const(*value, power);
        element_value = parser.add(element_value, value_two_pow);
    }

    parser.sub(element, element_value)
}

pub fn byte_decomposition_value<AP: AirParser, const N: usize>(
    element: AP::Var,
    values: &[AP::Var; N],
    parser: &mut AP,
) -> AP::Var {
    let two_powers: [_; N] = core::array::from_fn(|i| AP::Field::from_canonical_u32(1 << (8 * i)));

    let mut element_value = parser.zero();
    for (value, power) in values.iter().zip(two_powers) {
        let value_two_pow = parser.mul_const(*value, power);
        element_value = parser.add(element_value, value_two_pow);
    }

    parser.sub(element, element_value)
}
