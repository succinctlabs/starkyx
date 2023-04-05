//! The layout for a single ECADD circuit
//!
//!

use std::sync::mpsc::Sender;

use plonky2_maybe_rayon::*;

use super::*;
use crate::arithmetic::arithmetic_stark::EmulatedCircuitLayout;
use crate::arithmetic::polynomial::Polynomial;
use crate::arithmetic::{Instruction, Register};

const MUL_WITNESS: usize = fpmul::TOTAL_WITNESS_COLUMNS;
const DEN_WITNESS: usize = den::TOTAL_WITNESS_COLUMNS;
const MULD_WITNESS: usize = muld::TOTAL_WITNESS_COLUMNS;
const QUAD_WITNESS: usize = quad::TOTAL_WITNESS_COLUMNS;

/// A circuit performing ECADD
///
///
#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct ECAddChip {
    x_1: Register,
    y_1: Register,
    x_2: Register,
    y_2: Register,
    x_3: Register,
    y_3: Register,
    X3N: QuadLayout,
    Y3N: QuadLayout,
    X1Y1: FpMulLayout,
    X2Y2: FpMulLayout,
    ALLXY: FpMulLayout,
    DXY: MulDLayout,
    X3DEN: DenLayout,
    Y3DEN: DenLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    x: Register,
    y: Register,
}

impl Point {
    pub const fn new(register: Register) -> Self {
        match register {
            Register::Local(index, _) => Self {
                x: Register::Local(index, N_LIMBS),
                y: Register::Local(index + N_LIMBS, N_LIMBS),
            },
            Register::Next(index, _) => Self {
                x: Register::Next(index, N_LIMBS),
                y: Register::Next(index + N_LIMBS, N_LIMBS),
            },
        }
    }

    pub const fn x(&self) -> Register {
        self.x
    }
    pub const fn y(&self) -> Register {
        self.y
    }
}

impl ECAddChip {
    #[allow(non_snake_case)]
    pub const fn new(P1: Point, P2: Point, P3: Point, witness_index: usize) -> Self {
        let x_1 = P1.x();
        let y_1 = P1.y();
        let x_2 = P2.x();
        let y_2 = P2.y();
        let x_3 = P3.x();
        let y_3 = P3.y();

        // Set the X3N opcode layout
        let mut index = witness_index;
        let x3n_output = Register::Local(index, N_LIMBS);
        index += N_LIMBS;
        let x3n_witness = Register::Local(index, QUAD_WITNESS);
        index += QUAD_WITNESS;
        let X3N = QuadLayout::new(x_1, y_2, x_2, y_1, x3n_output, x3n_witness);

        // Set the Y3N opcode layout
        let y3n_output = Register::Local(index, N_LIMBS);
        index += N_LIMBS;
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
        index += DEN_WITNESS;
        let X3DEN = DenLayout::new(x3n_output, dxy_output, true, x_3, den_x3_witness);

        // Set DENY3 layout
        let den_y3_witness = Register::Local(index, DEN_WITNESS);
        let Y3DEN = DenLayout::new(y3n_output, dxy_output, false, y_3, den_y3_witness);

        Self {
            x_1,
            y_1,
            x_2,
            y_2,
            x_3,
            y_3,
            X3N,
            Y3N,
            X1Y1,
            X2Y2,
            ALLXY,
            DXY,
            X3DEN,
            Y3DEN,
        }
    }

    pub const fn input_length() -> usize {
        4 * N_LIMBS
    }

    pub const fn output_length() -> usize {
        2 * N_LIMBS
    }

    pub const fn witness_length() -> usize {
        N_LIMBS
            + QUAD_WITNESS
            + N_LIMBS
            + QUAD_WITNESS
            + N_LIMBS
            + MUL_WITNESS
            + N_LIMBS
            + MUL_WITNESS
            + N_LIMBS
            + MUL_WITNESS
            + N_LIMBS
            + MULD_WITNESS
            + DEN_WITNESS
            + DEN_WITNESS
    }

    pub const fn operations(&self) -> [EdOpcodeLayout; 8] {
        [
            EdOpcodeLayout::Quad(self.X3N),
            EdOpcodeLayout::Quad(self.Y3N),
            EdOpcodeLayout::FpMul(self.X1Y1),
            EdOpcodeLayout::FpMul(self.X2Y2),
            EdOpcodeLayout::FpMul(self.ALLXY),
            EdOpcodeLayout::MULD(self.DXY),
            EdOpcodeLayout::DEN(self.X3DEN),
            EdOpcodeLayout::DEN(self.Y3DEN),
        ]
    }

    pub const fn operations_with_input(
        &self,
        input: WriteInputLayout,
    ) -> [EpOpcodewithInputLayout; 9] {
        [
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::Quad(self.X3N)),
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::Quad(self.Y3N)),
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::FpMul(self.X1Y1)),
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::FpMul(self.X2Y2)),
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::FpMul(self.ALLXY)),
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::MULD(self.DXY)),
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::DEN(self.X3DEN)),
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::DEN(self.Y3DEN)),
            EpOpcodewithInputLayout::Input(input),
        ]
    }

    #[inline]
    pub const fn x3n_index() -> usize {
        0
    }

    #[inline]
    pub const fn y3n_index() -> usize {
        1
    }

    #[inline]
    pub const fn x1y1_index() -> usize {
        2
    }

    #[inline]
    pub const fn x2y2_index() -> usize {
        3
    }

    #[inline]
    pub const fn all_xy_index() -> usize {
        4
    }

    #[inline]
    pub const fn dxy_index() -> usize {
        5
    }

    #[inline]
    pub const fn x3den_index() -> usize {
        6
    }

    #[inline]
    pub const fn y3den_index() -> usize {
        7
    }

    #[inline]
    pub const fn input_index() -> usize {
        8
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ECAddInstruction;

impl ECAddInstruction {
    fn generate_trace<F: RichField + Extendable<D>, const D: usize>(
        x_1: &BigUint,
        y_1: &BigUint,
        x_2: &BigUint,
        y_2: &BigUint,
        pc: usize,
        tx: Sender<(usize, usize, Vec<F>)>,
    ) {
        let p_x_1 = Polynomial::<F>::from_biguint_field(&x_1, 16, N_LIMBS);
        let p_y_1 = Polynomial::<F>::from_biguint_field(&y_1, 16, N_LIMBS);
        let p_x_2 = Polynomial::<F>::from_biguint_field(&x_2, 16, N_LIMBS);
        let p_y_2 = Polynomial::<F>::from_biguint_field(&y_2, 16, N_LIMBS);
    }

    fn generate_trace_with_input<F: RichField + Extendable<D>, const D: usize>(
        x_1: &BigUint,
        y_1: &BigUint,
        x_2: &BigUint,
        y_2: &BigUint,
        pc: usize,
        tx: Sender<(usize, usize, Vec<F>)>,
    ) {
        // Write input to trace
        let p_x_1 = Polynomial::<F>::from_biguint_field(&x_1, 16, N_LIMBS);
        let p_y_1 = Polynomial::<F>::from_biguint_field(&y_1, 16, N_LIMBS);
        let p_x_2 = Polynomial::<F>::from_biguint_field(&x_2, 16, N_LIMBS);
        let p_y_2 = Polynomial::<F>::from_biguint_field(&y_2, 16, N_LIMBS);

        let mut input = p_x_1.as_slice().to_vec();
        input.extend_from_slice(p_y_1.as_slice());
        input.extend_from_slice(p_x_2.as_slice());
        input.extend_from_slice(p_y_2.as_slice());

        tx.send((pc, ECAddChip::input_index(), input)).unwrap();

        // calculate x3n and write to trace
        let x3n_op = EdOpcode::Quad(x_1.clone(), y_2.clone(), x_2.clone(), y_1.clone());
        let (x3n_trace, x3n): (Vec<F>, BigUint) = x3n_op.generate_trace_row();
        tx.send((pc, ECAddChip::x3n_index(), x3n_trace)).unwrap();

        // calculate y3n and write to trace
        let y3n_op = EdOpcode::Quad(y_1.clone(), y_2.clone(), x_1.clone(), x_2.clone());
        let (y3n_trace, y3n): (Vec<F>, BigUint) = y3n_op.generate_trace_row();
        tx.send((pc, ECAddChip::y3n_index(), y3n_trace)).unwrap();

        // calculate x1y1 and write to trace
        let x1y1_op = EdOpcode::FpMul(x_1.clone(), y_1.clone());
        let (x1y1_trace, x1y1): (Vec<F>, BigUint) = x1y1_op.generate_trace_row();
        tx.send((pc, ECAddChip::x1y1_index(), x1y1_trace)).unwrap();

        // calculate x2y2 and write to trace
        let x2y2_op = EdOpcode::FpMul(x_2.clone(), y_2.clone());
        let (x2y2_trace, x2y2): (Vec<F>, BigUint) = x2y2_op.generate_trace_row();
        tx.send((pc, ECAddChip::x2y2_index(), x2y2_trace)).unwrap();

        // calculate all_xy and write to trace
        let all_xy_op = EdOpcode::FpMul(x1y1, x2y2);
        let (all_xy_trace, all_xy): (Vec<F>, BigUint) = all_xy_op.generate_trace_row();
        tx.send((pc, ECAddChip::all_xy_index(), all_xy_trace))
            .unwrap();

        // calculate dxy and write to trace
        let dxy_op = EdOpcode::MULD(all_xy);
        let (dxy_trace, dxy): (Vec<F>, BigUint) = dxy_op.generate_trace_row();
        tx.send((pc, ECAddChip::dxy_index(), dxy_trace)).unwrap();

        // Calculate x_3 and write to trace
        let x3den_op = EdOpcode::DEN(x3n, dxy.clone(), true);
        let (x3den_trace, _): (Vec<F>, BigUint) = x3den_op.generate_trace_row();
        tx.send((pc, ECAddChip::x3den_index(), x3den_trace))
            .unwrap();

        // Calculate y_3 and write to trace
        let y3den_op = EdOpcode::DEN(y3n, dxy, false);
        let (y3den_trace, _): (Vec<F>, BigUint) = y3den_op.generate_trace_row();
        tx.send((pc, ECAddChip::y3den_index(), y3den_trace))
            .unwrap();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimpleRowEcAddCircuit;

#[derive(Debug, Clone)]
pub struct SimpleRowEcAddInstruction {
    pub x_1: BigUint,
    pub y_1: BigUint,
    pub x_2: BigUint,
    pub y_2: BigUint,
}

impl SimpleRowEcAddCircuit {
    pub const CHIP: ECAddChip = ECAddChip::new(
        Point::new(Register::Local(0, 2 * N_LIMBS)),
        Point::new(Register::Local(2 * N_LIMBS, 2 * N_LIMBS)),
        Point::new(Register::Local(4 * N_LIMBS, 2 * N_LIMBS)),
        6 * N_LIMBS,
    );

    pub const INPUT: WriteInputLayout = WriteInputLayout::new(Register::Local(0, 4 * N_LIMBS));

    pub const OPERATIONS: [EpOpcodewithInputLayout; 9] =
        Self::CHIP.operations_with_input(Self::INPUT);
}

impl<F: RichField + Extendable<D>, const D: usize> EmulatedCircuitLayout<F, D, 9>
    for SimpleRowEcAddCircuit
{
    const PUBLIC_INPUTS: usize = 0;
    const NUM_ARITHMETIC_COLUMNS: usize =
        ECAddChip::input_length() + ECAddChip::output_length() + ECAddChip::witness_length();
    const ENTRY_COLUMN: usize = 0;
    const TABLE_INDEX: usize = <Self as EmulatedCircuitLayout<F, D, 9>>::NUM_ARITHMETIC_COLUMNS;

    type Layouts = EpOpcodewithInputLayout;
    const OPERATIONS: [Self::Layouts; 9] = Self::OPERATIONS;
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<SimpleRowEcAddCircuit, F, D, 9>
    for SimpleRowEcAddInstruction
{
    fn generate_trace(self, pc: usize, tx: Sender<(usize, usize, Vec<F>)>) {
        //rayon::spawn(move || {
        ECAddInstruction::generate_trace_with_input(
            &self.x_1, &self.y_1, &self.x_2, &self.y_2, pc, tx,
        );
        //});
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;

    use super::*;
    use crate::arithmetic::arithmetic_stark::ArithmeticStark;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_simple_row_ecadd() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticStark<SimpleRowEcAddCircuit, 9, F, D>;

        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let b_x = BigUint::from_str_radix(
            "15112221349535400772501151409588531511454012693041857206046113283949847762202",
            10,
        )
        .unwrap();
        let b_y = BigUint::from_str_radix(
            "46316835694926478169428394003475163141307993866256225615783033603165251855960",
            10,
        )
        .unwrap();

        let mut add_instructions = Vec::new();
        for _ in 0..num_rows {
            let instruction = SimpleRowEcAddInstruction {
                x_1: b_x.clone(),
                y_1: b_y.clone(),
                x_2: b_x.clone(),
                y_2: b_y.clone(),
            };
            add_instructions.push(instruction);
        }

        //let mut rng = rand::thread_rng();
        let stark = S::new();
        let trace = stark.generate_trace(add_instructions);

        // Verify proof as a stark
        let proof =
            prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default()).unwrap();
        verify_stark_proof(stark, proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof =
            add_virtual_stark_proof_with_pis(&mut recursive_builder, stark, &config, degree_bits);

        recursive_builder.print_gate_counts(0);

        let mut rec_pw = PartialWitness::new();
        set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);

        verify_stark_proof_circuit::<F, C, S, D>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );

        let recursive_data = recursive_builder.build::<C>();

        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut timing,
        )
        .unwrap();

        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
