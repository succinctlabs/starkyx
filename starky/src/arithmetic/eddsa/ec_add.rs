//! The layout for a single ECADD circuit
//! 
//! 

use core::ops::Mul;

use crate::arithmetic::{arithmetic_stark::EmulatedCircuitLayout, Register};

use super::*;

const MUL_WITNESS : usize = fpmul::TOTAL_WITNESS_COLUMNS;
const DEN_WITNESS : usize = den::TOTAL_WITNESS_COLUMNS;
const MULD_WITNESS : usize = muld::TOTAL_WITNESS_COLUMNS;
const QUAD_WITNESS :usize = quad::TOTAL_WITNESS_COLUMNS;


/// A circuit performing ECADD 
/// 
/// 
#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct ECAddChip {
    x_1 : Register,
    y_1 : Register,
    x_2 : Register,
    y_2 : Register,
    x_3 : Register,
    y_3: Register,
    X3N : QuadLayout,
    Y3N : QuadLayout,
    X1Y1 : FpMulLayout,
    X2Y2 : FpMulLayout,
    ALLXY : FpMulLayout,
    DXY : MulDLayout,
    X3DEN : DenLayout,
    Y3DEN : DenLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct Point{ 
    x : Register,
    y : Register
}

impl Point {
    pub const fn new(register : Register) -> Self {
        match register {
            Register::Local(index, _) => Self { x: Register::Local(index, N_LIMBS), y: Register::Local(index + N_LIMBS, N_LIMBS) },
            Register::Next(index, _) => Self { x: Register::Next(index, N_LIMBS), y: Register::Next(index + N_LIMBS, N_LIMBS) }
        }
    }

    pub const fn x(&self) -> Register {
        self.x
    }
    pub const fn y(&self) -> Register {
        self.y
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ECAddWitness(Register);

impl ECAddWitness {

}

impl ECAddChip {

    #[allow(non_snake_case)]
    pub const fn new(P1 : Point, P2 : Point, P3: Point, witness_index : usize) -> Self {
        let x_1 = P1.x();
        let y_1 = P1.y();
        let x_2 = P2.x();
        let y_2 = P2.y();
        let x_3 = P3.x();
        let y_3 = P3.y();

        // Set the X3N opcode layout
        let mut index = witness_index;
        let x3n_output = Register::Local(index, N_LIMBS);
        index+=N_LIMBS;
        let x3n_witness = Register::Local(index, QUAD_WITNESS);
        index += QUAD_WITNESS;
        let X3N = QuadLayout::new(x_1, y_2, x_2, y_1, x3n_output, x3n_witness);

        // Set the Y3N opcode layout
        let y3n_output = Register::Local(index, N_LIMBS);
        index+=N_LIMBS;
        let y3n_witness = Register::Local(index, QUAD_WITNESS);
        index += QUAD_WITNESS;
        let Y3N = QuadLayout::new(y_1, y_2, x_1, x_2, y3n_output, y3n_witness);

        // Set X1Y1 opcode layout
        let x1y1_output = Register::Local(index, N_LIMBS);
        index += N_LIMBS;
        let x1y1_witness = Register::Local(index, MUL_WITNESS);
        index += MUL_WITNESS;
        let X1Y1 = FpMulLayout::new(x_1, y_1, x1y1_output, x1y1_witness);

        // Set X2Y2 opcode layout
        let x2y2_output = Register::Local(index, N_LIMBS);
        index += N_LIMBS;
        let x2y2_witness = Register::Local(index, MUL_WITNESS);
        index += MUL_WITNESS;
        let X2Y2 = FpMulLayout::new(x_2, y_2, x2y2_output, x2y2_witness);

        // Set the ALLXY (which is x1y1x2y2) layout
        let all_xy_output = Register::Local(index, N_LIMBS);
        index += N_LIMBS;
        let all_xy_witness = Register::Local(index, MUL_WITNESS);
        index += MUL_WITNESS;
        let ALLXY = FpMulLayout::new(x1y1_output, x2y2_output, all_xy_output, all_xy_witness);

        // Set DXY (which is d * x1y1x2y2) layout
        let dxy_output = Register::Local(index, N_LIMBS);
        index += N_LIMBS;
        let dxy_witness = Register::Local(index, MULD_WITNESS);
        index += MULD_WITNESS;
        let DXY = MulDLayout::new(all_xy_output, dxy_output, dxy_witness);

        // Set DENX3 layout
        let den_x3_witness = Register::Local(index, DEN_WITNESS);
        index+=DEN_WITNESS;
        let X3DEN = DenLayout::new(x3n_output, dxy_output, true, x_3, den_x3_witness);

        // Set DENY3 layout
        let den_y3_witness = Register::Local(index, DEN_WITNESS);
        index+=DEN_WITNESS;
        let Y3DEN = DenLayout::new(y3n_output, dxy_output, false, y_3, den_y3_witness);

        Self {x_1, y_1, x_2, y_2, x_3, y_3, X3N, Y3N, X1Y1, X2Y2, ALLXY, DXY, X3DEN, Y3DEN}
    }
} 

//const X1X2 : FpMulLayout = FpMulLayout::new(
//    Register::Local(0, N_LIMBS),
//);


/* 
impl<F: RichField + Extendable<D>, const D: usize> EmulatedCircuitLayout<F, D, 2> for ECADDCircuit {
    const PUBLIC_INPUTS: usize;
    const NUM_ARITHMETIC_COLUMNS: usize;
    const ENTRY_COLUMN: usize;
    const TABLE_INDEX: usize;

    type Layouts: OpcodeLayout<F, D>;
    const OPERATIONS: [Self::Layouts; N];
} 
*/

