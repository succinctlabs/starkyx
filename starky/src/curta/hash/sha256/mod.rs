//! This module implements the SHA256 hash function with an input message length of
//! 512 bits as an AIR-based recursive relation. The algorithm is summarized below.
//!
//! The pseudocode will make more sense if you reference `educational.rs`.
//!
//! <CONSTANTS>
//!
//! CONST_K_U32S[64]: First 32 bits of the fractional parts of the cube roots of the first 64 primes.
//! CONST_H_BITS[8][32]: First 32 bits of the fractional parts of the square roots of the first 8 primes.
//!
//! <REGISTERS>
//!
//! INPUT_U32S[16]: The input message of length 512 bits condensed into 16 u32's.
//! H_BITS[8][32]: The hash values array.
//! W_U32S[64]: The message schedule array.
//!
//! CHUNK_SELECTOR = 0: A selector bit that determines whether we are reading the first or the
//! second chunk of the message schedule array.
//!
//! MIXING_SELECTOR = 1: A selector bit that determines whether we are in the mixing selector.
//! COMPRESSING_SELECTOR = 0: A selector bit that determines whether we are in the compressing.
//! ACCUMULATION_SELECTOR = 0: A selector bit that determines whether we are accumulating.
//!
//! STEP_SELECTORS[48 + 64 + 1] = [0, ..., 0]
//!
//! J_SELECTORS[64] = [0, ..., 0]: Keeps track of the index 'j' and used to select from arrays.
//! J_SELECTORS[16] = 1
//!
//! <SELECTOR RANGE CHECKS>
//!
//! CHUNK_SELECTOR * (1 - CHUNK_SELECTOR) === 0
//! MIXING_SELECTOR * (1 - MIXING_SELECTOR) === 0
//! COMPRESSING_SELECTOR * (1 - COMPRESSING_SELECTOR) === 0
//! ACCUMULATION_SELECTOR * (1 - ACCUMULATION_SELECTOR) === 0
//! MIXING_SELECTOR + COMPRESSING_SELECTOR + ACCUMULATION_SELECTOR === 1
//! J_SELECTORS.SUM() === 1
//! for i in 0..64
//!     J_SELECTORS[i] * (1 - J_SELECTORS[i]) === 0
//!
//! <COORDINATOR>
//!
//! > Selects the correct message schedule array to use in the next 48 + 64 rows in the STARK.
//! W_CHUNK_1_U32S <== SOME_QUADRATIC_1(INPUT_U32S)
//! W_CHUNK_2_U32S <== SOME_QUADRATIC_2(INPUT_U32S)
//! W_U32S <== (1 - CHUNK_SELECTOR) * W_CHUNK_1_U32S + CHUNK_SELECTOR * W_CHUNK_1_U32S
//!
//! > Only increment if we are mixing or compressing.
//! J_INCREMENT_CONDITION <== MIXING_SELECTOR + COMPRESSING_SELECTOR
//!                         - MIXING_SELECTOR * COMPRESSING_SELECTOR
//!
//! > If we are in the accumulation stage it's time to reset j.
//! J_RESET_CONDITION <== ACCUMULATION_SELECTOR
//!
//! > If it's time to reset j, then reset j.
//! J_RESET_CONDITION * (J_SELECTOR[16].NEXT() - 1) === 0
//! for i in 0..64 except 16
//!     J_RESET_CONDITION * (J_SELECTOR[i].NEXT()) === 0
//!
//! > If it's not time to reset j, then increment j.
//! for i in 0..64
//!      (J_SELECTOR[i % 64].NEXT() - J_SELECTOR[(i - 1) % 64]) * (1 - J_RESET_CONDITION) === 0
//!
//! > Conditions for transitioning between phases.
//! MIXING_TO_COMPRESSING_CONDITION <== J_SELECTOR[63] * MIXING_SELECTOR
//! COMPRESSING_TO_ACCUMULATING_CONDITION <== J_SELECTOR[63] * COMPRESSING_SELECTOR
//! ACCUMULATING_TO_MIXING_CONDITION <== ACCUMULATION_SELECTOR
//!
//! > Transition between mixing to compressing.
//! COMPRESSSING_SELECTOR.NEXT() <== MIXING_TO_COMPRESSING
//! MIXING_SELECTOR.NEXT() <== MIXING_TO_COMPRESSING * 0 + (1 - MIXING_TO_COMPRESSING) * MIXING_SELECTOR
//!
//! > Transition between compressing to accumulating.
//! ACCUMULATION_SELECTOR.NEXT() <== COMPRESSING_TO_ACCUMULATING
//! COMPRESSING_SELECTOR.NEXT() <== COMPRESSING_TO_ACCUMULATING * 0 + (1 - COMPRESSING_TO_ACCUMULATING) * COMPRESSING_SELECTOR
//!
//! > Transition between accumulating to mixing.
//! MIXING_SELECTOR.NEXT() <== ACCUMULATING_TO_MIXING
//! ACCUMULATION_SELECTOR.NEXT() <== ACCUMULATING_TO_MIXING * 0 + (1 - ACCUMULATING_TO_MIXING) * ACCUMULATION_SELECTOR
//!
//! <MIXING>
//!
//! W_J_MINUS_15[32] <== J_SELECTOR.ROTATE(-15).DOT(W_U32).BITS()
//! W_J_MINUS_2_[32] <== J_SELECTOR.ROTATE(-2).DOT(W_U32).BITS()
//!
//! S0_WITNESS[32] <== W_J_MINUS_15.ROTATE(7).XOR(W_J_MINUS_15.ROTATE(18))
//! S0[32] <== S0_WITNESS[32].XOR(W_J_MINUS_15.SHR(3))
//!
//! S1_WITNESS[32] <== W_J_MINUS_2.ROTATE(17).XOR(W_J_MINUS_2.ROTATE(19))
//! S1[32] <== S1_WITNESS[32].XOR(W_J_MINUS_2.SHR(10))
//!
//! WARNING: OVERFLOW ISSUES WITH SMALL PROBABILITY?
//! W_U32.NEXT().DOT(J_SELECTOR) <== (1 - MIXING_PHASE_SELECTOR) * [W_J_MINUS_16.U32()
//!                                 + S0.U32()
//!                                 + W_J_MINUS_7.U32()
//!                                 + S1.U32()]
//!
//! <COMPRESSING>
//!
//! > Just copy, don't allocate.
//! SA <== H[0]
//! SB <== H[1]
//! SC <== H[2]
//! SD <== H[3]
//! SE <== H[4]
//! SF <== H[5]
//! SG <== H[6]
//! SH <== H[7]
//!
//! S1_WITNESS[32] <== SE.ROTATE(6).XOR(SE.ROTATE(11))
//! S1[32] <== S1_WITNESS.XOR(SE.ROTATE(25))
//!
//! CH_WITNESS_1[32] <== SE.AND(SF)
//! CH_WITNESS_2[32] <== SE.NOT().AND(SG)
//! CH[32] <== CH_WITNESS_1.XOR(CH_WITNESS_2)
//!
//! TEMP1[32] <== SH.U32() + S1.U32() + CH.U32() + K_U32.DOT(J_SELECTOR) + W_U32.DOT(J_SELECTOR)
//!
//! S0_WITNESS[32] <== SA.ROTATE(2).XOR(SA.ROTATE(13))
//! S0[32] <== S0_WITNESS.XOR(SA.ROTATE(22))
//!
//! MAJ_WITNESS_1[32] <== SA.AND(SB)
//! MAJ_WITNESS_2[32] <== SA.AND(SC)
//! MAJ_WITNESS_3[32] <== SB.AND(SC)
//! MAJ_WITNESS_4[32] <== MAJ_WITNESS_1.XOR(MAJ_WITNESS_2)
//! MAJ_WITNESS[32] <== MAJ_WITNESS_4.XOR(MAJ_WITNESS_3)
//!
//! TEMP2[32] <== (S0.U32() + MAJ.U32()).BITS()
//!
//! SH.NEXT() <== SG
//! SG.NEXT() <== SF
//! SF.NEXT() <== SE
//! SE.NEXT() <== (SD + TEMP1.U32()).BITS()
//! SD.NEXT() <== SC
//! SC.NEXT() <== SB
//! SB.NEXT() <== SA
//! SA.NEXT() <== (TEMP1.U32() + TEMP2.U32()).BITS()
//!
//! <ACCUMULATING>
//!
//! H_BITS[0] <== (H_BITS[0].U32() + SA.U32().BITS()
//! H_BITS[1] <== (H_BITS[1].U32() + SB.U32().BITS()
//! H_BITS[2] <== (H_BITS[2].U32() + SC.U32().BITS()
//! H_BITS[3] <== (H_BITS[3].U32() + SD.U32().BITS()
//! H_BITS[4] <== (H_BITS[4].U32() + SE.U32().BITS()
//! H_BITS[5] <== (H_BITS[5].U32() + SF.U32().BITS()
//! H_BITS[6] <== (H_BITS[6].U32() + SG.U32().BITS()
//! H_BITS[7] <== (H_BITS[7].U32() + SH.U32().BITS()

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::hash::sha256::helper::usize_to_be_bits;
use crate::curta::register::{ArrayRegister, BitRegister, ElementRegister, Register};

pub mod educational;
pub mod helper;
pub mod reference;

pub struct Sha256Gadget {
    input: ArrayRegister<BitRegister>,
}

// First 32 bits of the fractional parts of the square roots of the first 8 primes.
// Reference: https://en.wikipedia.org/wiki/SHA-2
const H: [usize; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

// First 32 bits of the fractional parts of the cube roots of the first 64 primes.
// Reference: https://en.wikipedia.org/wiki/SHA-2
const K: [usize; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const U64_BIT_LENGTH: usize = 64;
const SHA256_INPUT_LENGTH: usize = 512;
const SHA256_DIGEST_LENGTH: usize = 256;

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    #[allow(non_snake_case)]
    pub fn sha256(&mut self) {
        // The input message of length 512 bits condensed into 16 u32s.
        let input_u32s = self.alloc_array::<ElementRegister>(16);
    }
}

mod tests {
    use num::bigint::RandBigInt;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{StarkParameters, TestStark};
    use crate::curta::constraint::expression::ArithmeticExpression;
    use crate::curta::hash::sha256::helper::{rotate, shr, xor2};
    use crate::curta::instruction::InstructionSet;
    use crate::curta::parameters::ed25519::{Ed25519, Ed25519BaseField};
    use crate::curta::register::RegisterSerializable;
    use crate::curta::trace::trace;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    pub struct Sha256Test;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for Sha256Test {
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 385;
        type Instruction = InstructionSet<Ed25519BaseField>;
    }

    #[test]
    fn test_sha256() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519;
        type S = TestStark<Sha256Test, F, D>;
        let _ = env_logger::builder().is_test(true).try_init();

        // Build the stark.
        let mut builder = StarkBuilder::<Sha256Test, F, D>::new();

        const NB_MIXING_STEPS: usize = 48;
        const NB_COMPRESS_STEPS: usize = 64;
        const NB_ACCUMULATE_STEPS: usize = 1;
        const NB_STEPS: usize = NB_MIXING_STEPS + NB_COMPRESS_STEPS + NB_ACCUMULATE_STEPS;

        let step_selectors = builder.alloc_array::<BitRegister>(NB_STEPS);

        // Constrain step_selectors.sum() == 1.
        builder.constrain(step_selectors.expr().sum() - F::ONE);

        // Constrain step_selectors[i % NB_STEPS] == step_selectors[(i + 1) % NB_STEPS].next().
        for i in 0..NB_STEPS {
            builder.constrain_transition(
                step_selectors.get(i % NB_STEPS).expr()
                    - step_selectors.get((i + 1) % NB_STEPS).next().expr(),
            )
        }

        // The input message of length 512 bits condensed into 16 u32s.
        let input = builder.alloc_array::<ElementRegister>(16);
        let w_chunk_1 = builder.alloc_array::<ElementRegister>(64);
        for j in 0..16 {
            builder.constrain(
                step_selectors.get(0).expr() * (w_chunk_1.get(j).expr() - input.get(j).expr()),
            );
        }

        // Constrain w[j-15] and w[j-2] for j in 16..64.
        let w_j_minus_15_bits = builder.alloc_array::<BitRegister>(32);
        let w_j_minus_2_bits = builder.alloc_array::<BitRegister>(32);
        for j in 16..64 {
            let step = j - 16;
            builder.constrain(
                step_selectors.get(step).expr()
                    * (w_j_minus_15_bits.expr().be_sum() - w_chunk_1.get(j - 15).expr()),
            );
            builder.constrain(
                step_selectors.get(step).expr()
                    * (w_j_minus_2_bits.expr().be_sum() - w_chunk_1.get(j - 2).expr()),
            );
        }

        fn u32_xor2_constraint(
            a: ArithmeticExpression<F, D>,
            b: ArithmeticExpression<F, D>,
            c: ArithmeticExpression<F, D>,
        ) -> ArithmeticExpression<F, D> {
            c - (a.clone() + b.clone() - a * b * F::TWO)
        }

        // s0 = xor3(rotate(w[j - 15], 7), rotate(w[j - 15], 18), shr(w[j - 15], 3))
        let w_j_minus_15_rotated_7_bits = w_j_minus_15_bits.expr().rotate(7);
        let w_j_minus_15_rotated_18_bits = w_j_minus_15_bits.expr().rotate(18);
        let w_j_minus_15_shifted_3_bits = w_j_minus_15_bits.expr().shr(3);
        let s0_witness = builder.alloc_array::<BitRegister>(32);
        let s0 = builder.alloc_array::<BitRegister>(32);
        builder.constrain(u32_xor2_constraint(
            w_j_minus_15_rotated_7_bits,
            w_j_minus_15_rotated_18_bits,
            s0_witness.expr(),
        ));
        builder.constrain(u32_xor2_constraint(
            s0_witness.expr(),
            w_j_minus_15_shifted_3_bits,
            s0.expr(),
        ));

        // s1 = xor3(rotate(w[j - 2], 17), rotate(w[j - 2], 19), shr(w[j - 2], 10))
        let w_j_minus_2_rotated_17_bits = w_j_minus_2_bits.expr().rotate(17);
        let w_j_minus_2_rotated_19_bits = w_j_minus_2_bits.expr().rotate(19);
        let w_j_minus_2_shifted_10_bits = w_j_minus_2_bits.expr().shr(10);
        let s1_witness = builder.alloc_array::<BitRegister>(32);
        let s1 = builder.alloc_array::<BitRegister>(32);
        builder.constrain(u32_xor2_constraint(
            w_j_minus_2_rotated_17_bits,
            w_j_minus_2_rotated_19_bits,
            s1_witness.expr(),
        ));
        builder.constrain(u32_xor2_constraint(
            s1_witness.expr(),
            w_j_minus_2_shifted_10_bits,
            s1.expr(),
        ));

        let (chip, spec) = builder.build();

        // Generate the trace.
        let mut timing = TimingTree::new("sha256", log::Level::Debug);
        let nb_rows = 2usize.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let mut input_gt = [0u32; 16];
        for i in 0..16 {
            input_gt[i] = i as u32;
        }

        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = input_gt[i];
        }

        fn u32_arr_to_field_array(x: &[u32]) -> Vec<F> {
            x.into_iter()
                .map(|x| F::from_canonical_u32(*x))
                .collect::<Vec<_>>()
        }

        fn u32_to_be_field_bits(x: u32) -> Vec<F> {
            usize_to_be_bits::<32>(x as usize)
                .into_iter()
                .map(|x| F::from_canonical_u32(x as u32))
                .collect::<Vec<_>>()
        }

        fn bits_to_field_bits<const L: usize>(x: [bool; L]) -> Vec<F> {
            x.into_iter()
                .map(|x| F::from_canonical_u32(x as u32))
                .collect::<Vec<_>>()
        }

        let trace = timed!(timing, "witness generation", {
            for i in 0..nb_rows {
                // Write input.
                handle.write_data_v2(i, input, u32_arr_to_field_array(&input_gt));

                // Write step selectors.
                let step = i % NB_STEPS;
                for j in 0..NB_STEPS {
                    handle.write_bit(i, j == step, &step_selectors.get(j))
                }

                // Write w.
                handle.write_data_v2(i, w_chunk_1, u32_arr_to_field_array(&w));

                let is_mixing = step < NB_MIXING_STEPS;
                if is_mixing {
                    let j = 16 + step;
                    let w15 = u32_to_be_field_bits(w[j - 15]);
                    handle.write_data_v2(i, w_j_minus_15_bits, w15);
                    let w2 = u32_to_be_field_bits(w[j - 2]);
                    handle.write_data_v2(i, w_j_minus_2_bits, w2);

                    let w_j_minus_15_val = usize_to_be_bits::<32>(w[j - 15] as usize);
                    let s0_witness_val =
                        xor2(rotate(w_j_minus_15_val, 7), rotate(w_j_minus_15_val, 18));
                    handle.write_data_v2(i, s0_witness, bits_to_field_bits(s0_witness_val));

                    let s0_value = xor2(s0_witness_val, shr(w_j_minus_15_val, 3));
                    handle.write_data_v2(i, s0, bits_to_field_bits(s0_value));

                    let w_j_minus_2_val = usize_to_be_bits::<32>(w[j - 2] as usize);
                    let s1_witness_val =
                        xor2(rotate(w_j_minus_2_val, 17), rotate(w_j_minus_2_val, 19));
                    handle.write_data_v2(i, s1_witness, bits_to_field_bits(s1_witness_val));

                    let s1_value = xor2(s1_witness_val, shr(w_j_minus_2_val, 10));
                    handle.write_data_v2(i, s1, bits_to_field_bits(s1_value));
                }
            }
            drop(handle);
            generator.generate_trace(&chip, nb_rows as usize).unwrap()
        });

        // Verify proof as a stark
        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);
        let proof = timed!(
            timing,
            "proof generation",
            prove::<F, C, S, D>(
                stark.clone(),
                &config,
                trace,
                [],
                &mut TimingTree::default(),
            )
            .unwrap()
        );
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // // Verify recursive proof in a circuit.
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);
        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof = add_virtual_stark_proof_with_pis(
            &mut recursive_builder,
            stark.clone(),
            &config,
            degree_bits,
        );
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
        let recursive_proof = timed!(
            timing,
            "generate recursive proof",
            plonky2::plonk::prover::prove(
                &recursive_data.prover_only,
                &recursive_data.common,
                rec_pw,
                &mut TimingTree::default(),
            )
            .unwrap()
        );
        recursive_data.verify(recursive_proof).unwrap();
        timing.print();
    }
}
