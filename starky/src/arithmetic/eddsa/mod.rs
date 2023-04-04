//! Stark circuits for the EdDSA signature scheme.
//!
//!

use num::{BigUint, Num, One};
use plonky2::field::types::Field;

pub mod den;
pub mod ec_add;
pub mod fpmul;
pub mod quad;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use quad::QuadLayout;

use self::fpmul::FpMulLayout;
use super::{ArithmeticParser, Opcode, OpcodeLayout, WriteInputLayout};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

// General use constants

pub const LIMB: u32 = 2u32.pow(16);
pub const N_LIMBS: usize = 16;
const WITNESS_OFFSET: usize = 1usize << 20; // Witness offset

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

#[allow(non_snake_case)]
#[inline]
pub fn D_iter<F: Field>() -> impl Iterator<Item = F> {
    ED.iter().map(|&x| F::from_canonical_u16(x))
}

/// Layoutds for the Opcodes that comprise any Edwards curve operation.
#[derive(Debug, Clone, Copy)]
pub enum EdOpcodeLayout {
    Quad(QuadLayout),
    FpMul(FpMulLayout),
    DEN,
}

/// The core Opcodes that comprise any Edwards curve operation.
#[derive(Debug, Clone)]
pub enum EdOpcode {
    /// Quad(x_1, x_2, x_3, x_4) = (x_1 * x_2 + x_3 * x_4) mod p
    Quad(BigUint, BigUint, BigUint, BigUint),

    // FpMul(x_1, x_2) = (x_1 * x_2) mod p
    FpMul(BigUint, BigUint),

    /// DEN(a, b, sign) = a * (1 + sign * b)^{-1}
    /// In fact, we prove that a = b * result in the circuit
    DEN(BigUint, BigUint, bool),
}

impl<F: RichField + Extendable<D>, const D: usize> OpcodeLayout<F, D> for EdOpcodeLayout {
    fn read_input(&self, trace_rows: &mut [Vec<F>], row_index: usize) -> Vec<F> {
        match self {
            EdOpcodeLayout::Quad(quad) => quad.read_input(trace_rows, row_index),
            EdOpcodeLayout::FpMul(fpmul) => vec![], //fpmul.read_input(trace_rows, row, row_index),
            _ => unimplemented!("Operation not supported"),
        }
    }
    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            EdOpcodeLayout::Quad(quad) => quad.assign_row(trace_rows, row, row_index),
            EdOpcodeLayout::FpMul(fpmul) => fpmul.assign_row(trace_rows, row, row_index),
            _ => unimplemented!("Operation not supported"),
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
            EdOpcodeLayout::FpMul(fpmul) => {
                ArithmeticParser::fpmul_packed_generic_constraints(*fpmul, vars, yield_constr)
            }
            _ => unimplemented!("Operation not supported"),
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
            EdOpcodeLayout::FpMul(fpmul) => {
                ArithmeticParser::fpmul_ext_constraints(*fpmul, builder, vars, yield_constr)
            }
            _ => unimplemented!("Operation not supported"),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Opcode<F, D> for EdOpcode {
    fn generate_trace_row(self) -> Vec<F> {
        match self {
            EdOpcode::Quad(a, b, c, d) => ArithmeticParser::quad_trace(a, b, c, d),
            EdOpcode::FpMul(a, b) => ArithmeticParser::fpmul_trace(a, b),
            _ => unimplemented!("Operation not supported"),
        }
    }
}

// Helper for testing
pub enum EpOpcodewithInputLayout {
    Ep(EdOpcodeLayout),
    Input(WriteInputLayout),
}

impl<F: RichField + Extendable<D>, const D: usize> OpcodeLayout<F, D> for EpOpcodewithInputLayout {
    fn read_input(&self, trace_rows: &mut [Vec<F>], row_index: usize) -> Vec<F> {
        match self {
            EpOpcodewithInputLayout::Ep(opcode) => opcode.read_input(trace_rows, row_index),
            EpOpcodewithInputLayout::Input(opcode) => opcode.read_input(trace_rows, row_index),
        }
    }
    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            EpOpcodewithInputLayout::Ep(opcode) => opcode.assign_row(trace_rows, row, row_index),
            EpOpcodewithInputLayout::Input(opcode) => opcode.assign_row(trace_rows, row, row_index),
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
            EpOpcodewithInputLayout::Ep(opcode) => {
                opcode.packed_generic_constraints(vars, yield_constr)
            }
            EpOpcodewithInputLayout::Input(opcode) => {
                opcode.packed_generic_constraints(vars, yield_constr)
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
            EpOpcodewithInputLayout::Ep(opcode) => {
                opcode.ext_circuit_constraints(builder, vars, yield_constr)
            }
            EpOpcodewithInputLayout::Input(opcode) => {
                opcode.ext_circuit_constraints(builder, vars, yield_constr)
            }
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    pub fn ed_opcode_trace(opcode: EdOpcode) -> Vec<F> {
        match opcode {
            EdOpcode::Quad(x1, x2, x3, x4) => Self::quad_trace(x1, x2, x3, x4),
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
