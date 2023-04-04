use num::{BigUint, Num, One};
use plonky2::field::types::Field;

pub mod denominator;
pub mod ec_add;
pub mod quad;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::{ArithmeticParser, Opcode, OpcodeLayout};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const LIMB: u32 = 2u32.pow(16);

pub const P: [u16; 16] = [
    65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
    65535, 65535, 32767,
];

pub const ED: [u16; 16] = [
    30883, 4953, 19914, 30187, 55467, 16705, 2637, 112, 59544, 30585, 16505, 36039, 65139, 11119,
    27886, 20995,
];

pub fn get_p() -> BigUint {
    (BigUint::one() << 255) - BigUint::from(19u32)
}

pub fn get_d() -> BigUint {
    BigUint::from_str_radix(
        "37095705934669439343138083508754565189542113879843219016388785533085940283555",
        10,
    )
    .unwrap()
}

#[allow(non_snake_case)]
#[inline]
pub fn P_iter<F: Field>() -> impl Iterator<Item = F> {
    P.iter().map(|&x| F::from_canonical_u16(x))
}

pub trait Ed25519Param<const N_LIMBS: usize> {
    type F: Field;
    const P: [Self::F; N_LIMBS];
    const D: [Self::F; N_LIMBS];
}

/// Layoutds for the Opcodes that comprise any Edwards curve operation.
#[derive(Debug, Clone, Copy)]
pub enum EdOpcodeLayout {
    Quad(quad::QuadLayout),
}

impl<F: RichField + Extendable<D>, const D: usize> OpcodeLayout<F, D> for EdOpcodeLayout {
    fn assign_row<T: Copy>(&self, trace_rows: &mut [Vec<T>], row: &mut [T], row_index: usize) {
        match self {
            EdOpcodeLayout::Quad(quad) => quad.assign_row(trace_rows, row, row_index),
        }
    }

    fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            EdOpcodeLayout::Quad(quad) => {
                ArithmeticParser::quad_packed_generic_constraints(*quad, vars, yield_constr)
            }
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            EdOpcodeLayout::Quad(quad) => {
                ArithmeticParser::quad_ext_constraints(*quad, builder, vars, yield_constr)
            }
        }
    }
}

/// The core Opcodes that comprise any Edwards curve operation.
#[derive(Debug, Clone)]
pub enum EdOpcode {
    /// Quad(x_1, x_2, x_3, x_4) = (x_1 * x_2 + x_3 * x_4) mod p
    Quad(BigUint, BigUint, BigUint, BigUint),

    /// DEN(x_1, x_2, y_1, y_2, sign) = 1 + sign * D * (x_1 * x_2 * y_1 * y_2) mod p
    DEN(BigUint, BigUint, BigUint, BigUint, bool),
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    pub fn ed_opcode_trace(opcode: EdOpcode) -> Vec<F> {
        match opcode {
            EdOpcode::Quad(x1, x2, x3, x4) => Self::quad_trace(x1, x2, x3, x4),
            _ => unimplemented!("Operation not supported"),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Opcode<F, D> for EdOpcode {
    fn generate_trace(self) -> Vec<F> {
        match self {
            EdOpcode::Quad(a, b, c, d) => ArithmeticParser::quad_trace(a, b, c, d),
            _ => unimplemented!("Operation not supported"),
        }
    }
}

#[cfg(test)]
mod tests {

    use num::{BigUint, Num};

    use super::*;
    use crate::arithmetic::util;

    #[test]
    fn check_p25519_value() {
        let p = BigUint::from(2u32).pow(255) - BigUint::from(19u32);
        let p_limbs = util::bigint_into_u16_digits(&p, 16);

        assert_eq!(p_limbs, P);
        assert_eq!(p, get_p());
    }

    #[test]
    fn check_d_value() {
        let d = BigUint::from_str_radix(
            "37095705934669439343138083508754565189542113879843219016388785533085940283555",
            10,
        )
        .unwrap();

        // check the value of d is correct
        let p = BigUint::from(2u32).pow(255) - BigUint::from(19u32);
        assert_eq!((121666u32 * &d + 121665u32) % &p, BigUint::from(0u32));
        let d_limbs = util::bigint_into_u16_digits(&d, 16);
        assert_eq!(d_limbs, ED);

        let d_from_limbs = util::digits_to_biguint(&ED);
        assert_eq!(d, d_from_limbs);
        assert_eq!(d, get_d());
    }
}
