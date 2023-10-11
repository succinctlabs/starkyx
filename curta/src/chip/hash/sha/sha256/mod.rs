pub mod builder_gadget;
pub mod generator;

use core::borrow::Borrow;

use serde::{Deserialize, Serialize};

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::cycle::Cycle;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::chip::table::bus::global::Bus;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::uint::operations::add::ByteArrayAdd;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::U32Register;
use crate::chip::uint::util::u32_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub type U32Value<T> = <U32Register as Register>::Value<T>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHA256Gadget {
    /// The input chunks processed into 16-words of U32 values
    pub public_word: ArrayRegister<U32Register>,
    /// The hash states at all 1024 rounds
    pub state: ArrayRegister<U32Register>,
    /// The window of 16 w-values
    pub w_window: ArrayRegister<U32Register>,
    /// Signifies when to reset the state to the initial hash
    pub end_bit: BitRegister,
    pub(crate) end_bits_public: ArrayRegister<BitRegister>,
    pub(crate) w_bit: BitRegister,
    pub(crate) initial_state: ArrayRegister<U32Register>,
    pub(crate) round_constant: U32Register,
    pub round_constants_public: ArrayRegister<U32Register>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHA256PublicData<T> {
    pub public_w: Vec<U32Value<T>>,
    pub hash_state: Vec<U32Value<T>>,
    pub end_bits: Vec<T>,
}

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INITIAL_HASH: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub fn first_hash_value<F: Field>() -> [[F; 4]; 8] {
    [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ]
    .map(u32::to_le_bytes)
    .map(|x| x.map(F::from_canonical_u8))
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl<L: AirParameters> AirBuilder<L> {
    pub fn process_sha_256_batch(
        &mut self,
        clk: &ElementRegister,
        bus: &mut Bus<CubicRegister, L::CubicParams>,
        bus_channel_idx: usize,
        operations: &mut ByteLookupOperations,
    ) -> SHA256Gadget
    where
        L::Instruction: U32Instructions,
    {
        // Registers to be written to
        let w_window = self.alloc_array::<U32Register>(17);
        let w_bit = self.alloc::<BitRegister>();
        let end_bit = self.alloc::<BitRegister>();
        let msg_array = self.alloc_array::<U32Register>(8);
        let round_constant = self.alloc::<U32Register>();
        let cycle_64 = self.cycle(6);

        // Public values
        let public_w = self.alloc_array_public::<U32Register>(16 * 1024);
        let initial_state = self.alloc_array_public::<U32Register>(8);
        let round_constants_public = self.alloc_array_public::<U32Register>(64);
        let hash_state = self.alloc_array_public::<U32Register>(8 * 1024);
        let end_bits_public = self.alloc_array_public::<BitRegister>(1024);

        // Get the w value from the bus
        let w_challenges = self.alloc_challenge_array::<CubicRegister>(U32Register::size_of() + 1);
        let clk_w =
            self.accumulate_expressions(&w_challenges, &[clk.expr(), w_window.get(0).expr()]);
        self.output_from_bus_filtered(bus_channel_idx, clk_w, w_bit);

        // Get hash state challenges
        let state_challenges =
            self.alloc_challenge_array::<CubicRegister>(U32Register::size_of() * 8 + 1);

        // Get a challenge for the end bit
        let end_bit_challenge = self.alloc_challenge_array::<CubicRegister>(2);

        // Put end bit values into the bus
        let clk_end_bit =
            self.accumulate_expressions(&end_bit_challenge, &[clk.expr(), end_bit.expr()]);

        // Put the end_bit in the bus at the end of each round
        self.input_to_bus_filtered(bus_channel_idx, clk_end_bit, cycle_64.end_bit);
        // Constrain all other values of end_bit to zero
        self.assert_expression_zero(end_bit.expr() * cycle_64.end_bit.not_expr());

        // Put public w values and hash state in the bus
        for i in 0..1024 {
            let state_digest = self.accumulate_public_expressions(
                &state_challenges,
                &[
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(
                        i * 64 + 63,
                    )),
                    hash_state.get_subarray(i * 8..i * 8 + 8).expr(),
                ],
            );
            bus.output_global_value(&state_digest);

            let bit_digest = self.accumulate_public_expressions(
                &end_bit_challenge,
                &[
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(
                        i * 64 + 63,
                    )),
                    end_bits_public.get(i).expr(),
                ],
            );
            bus.output_global_value(&bit_digest);

            for k in 0..16 {
                let w = public_w.get(i * 16 + k);
                let clk_expr =
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(i * 64 + k));
                let digest =
                    self.accumulate_public_expressions(&w_challenges, &[clk_expr, w.expr()]);
                bus.insert_global_value(&digest);
            }
        }

        // Put the round constant into the bus
        let round_constant_challenges =
            self.alloc_challenge_array::<CubicRegister>(U32Register::size_of() + 1);

        for k in 0..64 {
            let round_constant_public_input_digest = self.accumulate_public_expressions(
                &round_constant_challenges,
                &[
                    ArithmeticExpression::from_constant(
                        L::Field::from_canonical_u8(k as u8) - L::Field::from_canonical_u8(64),
                    ),
                    round_constants_public.get(k).expr(),
                ],
            );
            bus.insert_global_value(&round_constant_public_input_digest);

            let num_rows = 1 << 16;
            let round_constants_public_output_digest = self.accumulate_public_expressions(
                &round_constant_challenges,
                &[
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(
                        num_rows - 1 + k - 63,
                    )),
                    round_constants_public.get(k).expr(),
                ],
            );
            bus.output_global_value(&round_constants_public_output_digest);
        }

        let round_constant_output = self.accumulate_expressions(
            &round_constant_challenges,
            &[
                clk.expr() - L::Field::from_canonical_u8(64),
                round_constant.expr(),
            ],
        );
        self.output_from_bus(bus_channel_idx, round_constant_output);
        let round_constant_input = self.accumulate_expressions(
            &round_constant_challenges,
            &[clk.expr(), round_constant.expr()],
        );
        self.input_to_bus(bus_channel_idx, round_constant_input);

        // The premessage state
        self.sha_premessage(clk, &w_window, &w_bit, &cycle_64, operations);

        // Set the window values
        for i in 1..17 {
            self.set_to_expression_transition(&w_window.get(i).next(), w_window.get(i - 1).expr());
        }

        let hash = self.alloc_array::<U32Register>(8);
        for (h, init) in hash.iter().zip(initial_state.iter()) {
            self.set_to_expression_first_row(&h, init.expr());
        }
        // The SHA step phase
        let hash_next = self.sha_256_step(
            &hash,
            clk,
            &msg_array,
            &w_window,
            &initial_state,
            &round_constant,
            &cycle_64,
            &end_bit,
            operations,
        );

        // Connect hash to hash next depending on end_bit
        for i in 0..8 {
            self.set_to_expression_transition(
                &hash.get(i).next(),
                hash.get(i).expr() * cycle_64.end_bit.not_expr()
                    + msg_array.get(i).next().expr() * cycle_64.end_bit.expr(),
            );
        }

        let clk_hash_next =
            self.accumulate_expressions(&state_challenges, &[clk.expr(), hash_next.expr()]);
        self.input_to_bus_filtered(bus_channel_idx, clk_hash_next, cycle_64.end_bit);

        // Dummy operation because of an odd number of operations
        let dummy = self.alloc::<ByteRegister>();
        let dummy_range = ByteOperation::Range(dummy);
        self.set_byte_operation(&dummy_range, operations);

        SHA256Gadget {
            public_word: public_w,
            state: hash_state,
            end_bit,
            w_window,
            w_bit,
            initial_state,
            round_constant,
            round_constants_public,
            end_bits_public,
        }
    }

    fn sha_premessage(
        &mut self,
        clk: &ElementRegister,
        w_window: &ArrayRegister<U32Register>,
        w_bit: &BitRegister,
        cycle_64: &Cycle<L::Field>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: U32Instructions,
    {
        let cycle_16 = self.cycle(4);

        // Assert w_bit = 1 when start_bit = 1
        self.assert_expression_zero(w_bit.not_expr() * cycle_64.start_bit.expr());
        // Assert w_next = w whenever we are not at an end of a 16 loop
        self.assert_expression_zero_transition(
            (w_bit.expr() - w_bit.next().expr()) * cycle_16.end_bit.not_expr(),
        );
        // Assert w_next = 0 when we are at the end of a 16 loop but not a 64 loop
        self.assert_expression_zero_transition(
            w_bit.next().expr() * cycle_16.end_bit.expr() * cycle_64.end_bit.not_expr(),
        );

        // Calculate s_0 = w_i_minus_15.rotate_right(7) ^ w_i_minus_15.rotate_right(18) ^ (w_i_minus_15 >> 3);
        let w_i_minus_15_rotate_7 = self.bit_rotate_right(&w_window.get(15), 7, operations);
        let w_i_minus_15_rotate_18 = self.bit_rotate_right(&w_window.get(15), 18, operations);
        let w_i_minus_15_shr_3 = self.bit_shr(&w_window.get(15), 3, operations);

        let mut s_0 = self.bitwise_xor(&w_i_minus_15_rotate_7, &w_i_minus_15_rotate_18, operations);
        s_0 = self.bitwise_xor(&s_0, &w_i_minus_15_shr_3, operations);

        // Calculate s_1 = w_i_minus_2.rotate_right(17) ^ w_i_minus_2.rotate_right(19) ^ (w_i_minus_2 >> 10);
        let w_i_minus_2_rotate_17 = self.bit_rotate_right(&w_window.get(2), 17, operations);
        let w_i_minus_2_rotate_19 = self.bit_rotate_right(&w_window.get(2), 19, operations);
        let w_i_minus_2_shr_10 = self.bit_shr(&w_window.get(2), 10, operations);

        let mut s_1 = self.bitwise_xor(&w_i_minus_2_rotate_17, &w_i_minus_2_rotate_19, operations);
        s_1 = self.bitwise_xor(&s_1, &w_i_minus_2_shr_10, operations);

        // Calculate w_i = w_i_minus_16 + s_0 + w_i_minus_7 + s_1;
        let mut w_i = self.add_u32(&w_window.get(16), &s_0, operations);
        w_i = self.add_u32(&w_i, &w_window.get(7), operations);
        w_i = self.add_u32(&w_i, &s_1, operations);
        self.assert_expression_zero(w_bit.not_expr() * (w_i.expr() - w_window.get(0).expr()));
    }

    #[allow(clippy::too_many_arguments)]
    fn sha_256_step(
        &mut self,
        hash: &ArrayRegister<U32Register>,
        clk: &ElementRegister,
        msg: &ArrayRegister<U32Register>,
        w_window: &ArrayRegister<U32Register>,
        initial_state: &ArrayRegister<U32Register>,
        round_constant: &U32Register,
        cycle_64: &Cycle<L::Field>,
        end_bit: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) -> ArrayRegister<U32Register>
    where
        L::Instruction: U32Instructions,
    {
        // Initialize working variables
        let a = msg.get(0);
        let b = msg.get(1);
        let c = msg.get(2);
        let d = msg.get(3);
        let e = msg.get(4);
        let f = msg.get(5);
        let g = msg.get(6);
        let h = msg.get(7);

        for i in 0..8 {
            self.set_to_expression_first_row(&msg.get(i), initial_state.get(i).expr());
        }

        // Calculate sum_1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let e_rotate_6 = self.bit_rotate_right(&e, 6, operations);
        let e_rotate_11 = self.bit_rotate_right(&e, 11, operations);
        let e_rotate_25 = self.bit_rotate_right(&e, 25, operations);
        let mut sum_1 = self.bitwise_xor(&e_rotate_6, &e_rotate_11, operations);
        sum_1 = self.bitwise_xor(&sum_1, &e_rotate_25, operations);

        // Calculate ch = (e & f) ^ (!e & g);
        let e_and_f = self.bitwise_and(&e, &f, operations);
        let not_e = self.bitwise_not(&e, operations);
        let not_e_and_g = self.bitwise_and(&not_e, &g, operations);
        let ch = self.bitwise_xor(&e_and_f, &not_e_and_g, operations);

        // Calculate temp_1 = h + sum_1 +ch + round_constant + w;
        let mut temp_1 = self.add_u32(&h, &sum_1, operations);
        temp_1 = self.add_u32(&temp_1, &ch, operations);
        temp_1 = self.add_u32(&temp_1, round_constant, operations);
        temp_1 = self.add_u32(&temp_1, &w_window.get(0), operations);

        // Calculate sum_0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let a_rotate_2 = self.bit_rotate_right(&a, 2, operations);
        let a_rotate_13 = self.bit_rotate_right(&a, 13, operations);
        let a_rotate_22 = self.bit_rotate_right(&a, 22, operations);
        let mut sum_0 = self.bitwise_xor(&a_rotate_2, &a_rotate_13, operations);
        sum_0 = self.bitwise_xor(&sum_0, &a_rotate_22, operations);

        // Calculate maj = (a & b) ^ (a & c) ^ (b & c);
        let a_and_b = self.bitwise_and(&a, &b, operations);
        let a_and_c = self.bitwise_and(&a, &c, operations);
        let b_and_c = self.bitwise_and(&b, &c, operations);
        let mut maj = self.bitwise_xor(&a_and_b, &a_and_c, operations);
        maj = self.bitwise_xor(&maj, &b_and_c, operations);

        // Calculate temp_2 = sum_0 + maj;
        let temp_2 = self.add_u32(&sum_0, &maj, operations);

        // Calculate the next cycle values
        let a_next = self.add_u32(&temp_1, &temp_2, operations);
        let b_next = a;
        let c_next = b;
        let d_next = c;
        let e_next = self.add_u32(&d, &temp_1, operations);
        let f_next = e;
        let g_next = f;
        let h_next = g;

        let msg_next = [
            a_next, b_next, c_next, d_next, e_next, f_next, g_next, h_next,
        ];

        // Assign the hash values in the end of the round
        let hash_next = self.alloc_array::<U32Register>(8);
        for ((h, m_next), h_next) in hash.iter().zip(msg_next.iter()).zip(hash_next.iter()) {
            let carry = self.alloc::<BitRegister>();
            let add = ByteArrayAdd::<4>::new(h, *m_next, None, h_next, carry);
            self.register_instruction(add);

            for byte in h_next.to_le_bytes() {
                let result_range = ByteOperation::Range(byte);
                self.set_byte_operation(&result_range, operations);
            }
        }

        // Assign next values to the next row registers based on the cycle bit
        let bit = cycle_64.end_bit;
        for (((m, m_next), h_next), init) in msg
            .iter()
            .zip(msg_next.iter())
            .zip(hash_next.iter())
            .zip(initial_state.iter())
        {
            self.set_to_expression_transition(
                &m.next(),
                m_next.expr() * bit.not_expr()
                    + (h_next.expr() * end_bit.not_expr() + init.expr() * end_bit.expr())
                        * bit.expr(),
            );
        }

        hash_next
    }
}

impl SHA256Gadget {
    pub fn write<F: Field, I: IntoIterator>(
        &self,
        padded_messages: I,
        writer: &TraceWriter<F>,
    ) -> SHA256PublicData<F>
    where
        I::Item: Borrow<[u8]>,
    {
        let mut w_values = Vec::new();
        let mut end_bits_values = Vec::new();
        let mut hash_values = Vec::new();
        let mut public_w_values = Vec::new();

        padded_messages.into_iter().for_each(|padded_msg| {
            let padded_msg = padded_msg.borrow();
            let num_chunks = padded_msg.len() / 64;
            end_bits_values.extend_from_slice(&vec![F::ZERO; num_chunks - 1]);
            end_bits_values.push(F::ONE);

            let mut state = INITIAL_HASH;
            for chunk in padded_msg.chunks_exact(64) {
                let w_val = SHA256Gadget::process_inputs(chunk);
                public_w_values.extend(w_val[0..16].iter().map(|x| u32_to_le_field_bytes::<F>(*x)));
                state = SHA256Gadget::compress_round(state, &w_val, ROUND_CONSTANTS);
                w_values.extend_from_slice(&w_val.map(u32_to_le_field_bytes::<F>));
                hash_values.extend_from_slice(&state.map(u32_to_le_field_bytes::<F>));
            }
        });
        assert!(
            w_values.len() == 1024 * 64,
            "Padded messages lengths do not add up"
        );

        writer.write_array(
            &self.initial_state,
            INITIAL_HASH.map(u32_to_le_field_bytes),
            0,
        );
        writer.write_array(
            &self.round_constants_public,
            ROUND_CONSTANTS.map(u32_to_le_field_bytes),
            0,
        );
        writer.write_array(&self.state, &hash_values, 0);
        writer.write_array(&self.end_bits_public, &end_bits_values, 0);
        writer.write_array(&self.public_word, &public_w_values, 0);
        (0..1024).for_each(|i| {
            writer.write(&self.end_bit, &end_bits_values[i], i * 64 + 63);
            for j in 0..64 {
                let row = i * 64 + j;
                writer.write(
                    &self.round_constant,
                    &u32_to_le_field_bytes(ROUND_CONSTANTS[j]),
                    row,
                );
                writer.write(&self.w_window.get(0), &w_values[i * 64 + j], row);
                if j < 16 {
                    writer.write(&self.w_bit, &F::ONE, row);
                }
            }
        });

        SHA256PublicData {
            public_w: public_w_values,
            hash_state: hash_values,
            end_bits: end_bits_values,
        }
    }

    pub fn process_inputs(chunk: &[u8]) -> [u32; 64] {
        let chunk_u32 = chunk
            .chunks_exact(4)
            .map(|x| u32::from_be_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut w = [0u32; 64];

        w[..16].copy_from_slice(&chunk_u32[..16]);

        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        w
    }

    pub fn compress_round(hash: [u32; 8], w: &[u32; 64], round_constants: [u32; 64]) -> [u32; 8] {
        let mut msg = hash;
        for i in 0..64 {
            msg = SHA256Gadget::step(msg, w[i], round_constants[i]);
        }

        [
            hash[0].wrapping_add(msg[0]),
            hash[1].wrapping_add(msg[1]),
            hash[2].wrapping_add(msg[2]),
            hash[3].wrapping_add(msg[3]),
            hash[4].wrapping_add(msg[4]),
            hash[5].wrapping_add(msg[5]),
            hash[6].wrapping_add(msg[6]),
            hash[7].wrapping_add(msg[7]),
        ]
    }

    pub fn step(msg: [u32; 8], w_i: u32, round_constant: u32) -> [u32; 8] {
        let mut a = msg[0];
        let mut b = msg[1];
        let mut c = msg[2];
        let mut d = msg[3];
        let mut e = msg[4];
        let mut f = msg[5];
        let mut g = msg[6];
        let mut h = msg[7];

        let sum_1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp_1 = h
            .wrapping_add(sum_1)
            .wrapping_add(ch)
            .wrapping_add(round_constant)
            .wrapping_add(w_i);
        let sum_0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp_2 = sum_0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp_1);
        d = c;
        c = b;
        b = a;
        a = temp_1.wrapping_add(temp_2);

        [a, b, c, d, e, f, g, h]
    }

    pub fn pad(msg: &[u8]) -> Vec<u8> {
        let mut padded_msg = Vec::new();
        padded_msg.extend_from_slice(msg);
        padded_msg.push(1 << 7);

        // Find number of zeros
        let mdi = msg.len() % 64;
        assert!(mdi < 120);
        let padlen = if mdi < 56 { 55 - mdi } else { 119 - mdi };
        // Pad with zeros
        padded_msg.extend_from_slice(&vec![0u8; padlen]);

        // add length as 64 bit number
        let len = ((msg.len() * 8) as u64).to_be_bytes();
        padded_msg.extend_from_slice(&len);

        padded_msg
    }
}

#[cfg(test)]
mod tests {

    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use subtle_encoding::hex::decode;

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::uint::util::u32_to_le_field_bytes;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct SHA256Test;

    impl AirParameters for SHA256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_FREE_COLUMNS: usize = 745;
        const EXTENDED_COLUMNS: usize = 345;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
    }

    #[test]
    fn test_sha_256_stark() {
        type F = GoldilocksField;
        type L = SHA256Test;
        type SC = PoseidonGoldilocksStarkConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Sha256 test", log::Level::Debug);

        let mut builder = AirBuilder::<L>::new();
        let clk = builder.clock();

        let mut operations = builder.byte_operations();

        let mut bus = builder.new_bus();
        let channel_idx = bus.new_channel(&mut builder);

        let sha_gadget =
            builder.process_sha_256_batch(&clk, &mut bus, channel_idx, &mut operations);

        let byte_data = builder.register_byte_lookup(operations);
        builder.constrain_bus(bus);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;

        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let short_msg_1 = decode("").unwrap();
        let expected_digest_1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        // let short_msg_1 = b"plonky2".to_vec();
        // let expected_digest = "8943a85083f16e93dc92d6af455841daacdae5081aa3125b614a626df15461eb";

        let short_msg_2 = b"abc".to_vec();
        let expected_digest_2 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let long_msg = decode("243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917").unwrap();
        let expected_digest_long =
            "aca16131a2e4c4c49e656d35aac1f0e689b3151bb108fa6cf5bcc3ac08a09bf9";

        let messages = (0..256)
            .flat_map(|_| [short_msg_1.clone(), long_msg.clone(), short_msg_2.clone()])
            .collect::<Vec<_>>();
        let padded_messages = messages
            .iter()
            .map(|m| SHA256Gadget::pad(m))
            .collect::<Vec<_>>();

        let expected_digests: Vec<[u32; 8]> = (0..256)
            .flat_map(|_| [expected_digest_1, expected_digest_long, expected_digest_2])
            .map(|digest| {
                hex::decode(digest)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|x| u32::from_be_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(expected_digests.len(), padded_messages.len());

        let mut digest_iter = expected_digests.into_iter();
        timed!(timing, "Write the execusion trace", {
            byte_data.write_table_entries(&writer);
            sha_gadget.write(padded_messages, &writer);
            for i in 0..num_rows {
                writer.write_row_instructions(&generator.air_data, i);
                let end_bit = writer.read(&sha_gadget.end_bit, i);
                if end_bit == F::ONE {
                    let j = (i - 63) / 64;
                    let hash =
                        writer.read_array(&sha_gadget.state.get_subarray(j * 8..j * 8 + 8), 0);
                    let digest = digest_iter.next().unwrap();
                    assert_eq!(hash, digest.map(u32_to_le_field_bytes));
                }
            }
            let multiplicities = byte_data.get_multiplicities(&writer);
            writer.write_lookup_multiplicities(byte_data.multiplicities(), &[multiplicities]);
        });

        let public_inputs = writer.0.public.read().unwrap().clone();
        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        timed!(
            timing,
            "Stark proof and verify",
            test_starky(&stark, &config, &generator, &public_inputs)
        );

        // Generate recursive proof
        timed!(
            timing,
            "Recursive proof generation and verification",
            test_recursive_starky(stark, config, generator, &public_inputs)
        );

        timing.print();
    }
}
