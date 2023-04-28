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
use crate::curta::register::{ArrayRegister, BitRegister, Register};

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
    pub fn sha256<const SHA256_INPUT_LENGTH: usize>(&mut self, input: ArrayRegister<BitRegister>) {
        assert!(
            input.len() == SHA256_INPUT_LENGTH,
            "Input length does not match SHA256_INPUT_LENGTH"
        );

        // The length-encoded message length ("L + 1 + 64").
        let SEPERATOR_LENGTH: usize = 1;
        let U64_LENGTH: usize = 64;
        let ENCODED_MESSAGE_LENGTH: usize = SHA256_INPUT_LENGTH + SEPERATOR_LENGTH + U64_LENGTH;

        // The multiple of 512-bit padded message length. Padding length is "K".
        let REMAINDER_LENGTH: usize = ENCODED_MESSAGE_LENGTH % 512;
        let PADDING_LENGTH: usize = if REMAINDER_LENGTH == 0 {
            0
        } else {
            512 - REMAINDER_LENGTH
        };
        let PADDED_MESSAGE_LENGTH: usize = ENCODED_MESSAGE_LENGTH + PADDING_LENGTH;

        // Initialization of core variables.
        let padded_message = self.alloc_array::<BitRegister>(PADDED_MESSAGE_LENGTH);

        // Note: THIS IS INEFFICIENT. We don't need to COPY, we can just READ.

        // Begin with the original message of length "L".
        for i in 0..SHA256_INPUT_LENGTH {
            let constraint = padded_message.get(i).expr::<F, D>() - input.get(i).expr::<F, D>();
            self.assert_expression_zero(constraint);
        }

        // Append a single '1' bit.
        let constraint = padded_message.get(SHA256_INPUT_LENGTH).expr() - F::ONE;
        self.assert_expression_zero(constraint);

        // Append L as a 64-bit big-endian integer.
        let sha256_input_length_bits = usize_to_be_bits::<U64_BIT_LENGTH>(SHA256_INPUT_LENGTH);
        for i in 0..U64_BIT_LENGTH {
            let zero_or_one = if sha256_input_length_bits[i] {
                F::ONE
            } else {
                F::ZERO
            };
            let constraint = padded_message
                .get(SHA256_INPUT_LENGTH + i + 1 + PADDING_LENGTH)
                .expr()
                - zero_or_one;
            self.assert_expression_zero(constraint);
        }

        // At this point, the padded message should be of the following form.
        //      <message of length L> 1 <K zeros> <L as 64 bit integer>
        // Now, we will process the padded message in 512 bit chunks and begin referring to the
        // padded message as "message".
        const SHA256_CHUNK_LENGTH: usize = 512;
        const SHA256_WORD_LENGTH: usize = 32;
        const SHA256_MESSAGE_SCHEDULE_ARRAY_LENGTH: usize = 64;

        let message = padded_message;
        let num_chunks = message.len() / SHA256_CHUNK_LENGTH;

        let mut h = H
            .into_iter()
            .map(|x| usize_to_be_bits::<SHA256_WORD_LENGTH>(x))
            .collect::<Vec<_>>();
        let k = K
            .into_iter()
            .map(|x| usize_to_be_bits::<SHA256_WORD_LENGTH>(x))
            .collect::<Vec<_>>();

        // For simplicity, we should just load all 1024 bits of the input into a register.
        // Then, we copy the correct bits into the message schedule array. However, we don't need to
        // allocate, it can just be a copy of the bits.

        // Register Layout
        // > Note that the padded_message[512:1024] are all CONSTANTS.
        // > Note that we only read the past 16 entries of w in the mixing step. On the next row,
        // > we can simply read the next set of entires?

        for i in 0..num_chunks {
            // The 64-entry message schedule array of 32-bit words.
            let w = (0..SHA256_MESSAGE_SCHEDULE_ARRAY_LENGTH)
                .into_iter()
                .map(|_| self.alloc_array::<BitRegister>(SHA256_WORD_LENGTH))
                .collect::<Vec<_>>();

            // Copy chunk into first 16 words w[0..15] of the message schedule array.
            let chunk_offset = i * SHA256_CHUNK_LENGTH;
            for j in 0..16 {
                let word_offset = j * 32;
                for k in 0..32 {
                    let constraint =
                        w[j].get(k).expr() - message.get(chunk_offset + word_offset + k).expr();
                    self.assert_expression_zero(constraint);
                }
            }
        }
    }
}
