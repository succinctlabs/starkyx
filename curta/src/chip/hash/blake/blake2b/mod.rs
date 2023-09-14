pub mod builder_gadget;
pub mod generator;

use core::borrow::Borrow;

use itertools::Itertools;

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::chip::table::bus::global::Bus;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub type U64Value<T> = <U64Register as Register>::Value<T>;

#[derive(Debug, Clone)]
pub struct BLAKE2BGadget {
    pub t: U64Register, // Need to constrain
    pub m: ArrayRegister<U64Register>,
    pub first_block_bit: BitRegister,    // need to constrain
    pub last_block_bit: BitRegister,     // need to constrain
    pub cycle_12_start_bit: BitRegister, // need to constrain
    pub cycle_12_end_bit: BitRegister,   // need to constrain

    // Public values
    pub msg_chunks: ArrayRegister<U64Register>,
    pub initial_hash: ArrayRegister<U64Register>,
    pub initial_hash_compress: ArrayRegister<U64Register>,
    pub hash_state: ArrayRegister<U64Register>,
    pub inversion_const: U64Register,
}

#[derive(Debug, Clone)]
pub struct BLAKE2BPublicData<T> {
    pub hash_state: Vec<U64Value<T>>,
    pub end_bits: Vec<T>,
}

const INITIAL_HASH: [u64; 8] = [
    0x6a09e667f2bdc928,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

// Note that for this blake2b implementation, we don't support a key input and
// we assume that the output is 32 bytes
// So that means the initial hash entry to be
// 0x6a09e667f3bcc908 xor 0x01010020
const INITIAL_HASH_COMPRESS: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const INVERSION_CONST: u64 = 0xFFFFFFFFFFFFFFFF;

const SIGMA_LEN: usize = 10;
const SIGMA: [[usize; 16]; SIGMA_LEN] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

impl<L: AirParameters> AirBuilder<L> {
    pub fn process_blake2b(
        &mut self,
        clk: &ElementRegister,
        bus: &mut Bus<L::CubicParams>,
        bus_channel_idx: usize,
        operations: &mut ByteLookupOperations,
    ) -> BLAKE2BGadget
    where
        L::Instruction: U32Instructions,
    {
        // Registers to be written to
        let t = self.alloc::<U64Register>(); // need to constrain
        let m = self.alloc_array::<U64Register>(16);
        let h_input = self.alloc_array::<U64Register>(8);
        let h_output = self.alloc_array::<U64Register>(8);
        let first_block_bit = self.alloc::<BitRegister>(); // need to constrain
        let last_block_bit = self.alloc::<BitRegister>();

        // TODO:  Need to write to these registers and constraint them
        let cycle_12_start_bit = self.alloc::<BitRegister>(); // need to constrain
        let cycle_12_end_bit = self.alloc::<BitRegister>(); // need to constrain

        // Public values
        let msg_chunks = self.alloc_array_public::<U64Register>(16 * 512);
        let initial_hash = self.alloc_array_public::<U64Register>(8);
        let initial_hash_compress = self.alloc_array_public::<U64Register>(8);
        let hash_state = self.alloc_array_public::<U64Register>(8 * 512);
        let inversion_const = self.alloc_public::<U64Register>();

        // Get message chunk challenges
        let message_chunk_challenges =
            self.alloc_challenge_array::<CubicRegister>(U64Register::size_of() * 16 + 1);

        // Get hash state challenges
        let state_challenges =
            self.alloc_challenge_array::<CubicRegister>(U64Register::size_of() * 8 + 1);

        // Put public hash state, end_bits, and all the msg chunk permutations into the bus
        for i in 0..512 {
            let state_digest = self.accumulate_public_expressions(
                &state_challenges,
                &[
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(
                        i * 12 + 11,
                    )),
                    hash_state.get_subarray(i * 8..i * 8 + 8).expr(),
                ],
            );

            bus.output_global_value(&state_digest);

            for k in 0..12 {
                let row = i * 12 + k;
                let sigma = SIGMA[k % SIGMA_LEN];
                let mut values = Vec::new();

                values.push(ArithmeticExpression::from_constant(
                    L::Field::from_canonical_usize(row),
                ));

                for index in sigma.iter() {
                    values.push(msg_chunks.get(i * 16 + index).expr());
                }

                let msg_digest = self
                    .accumulate_public_expressions(&message_chunk_challenges, values.as_slice());

                bus.output_global_value(&msg_digest);
            }
        }

        let clk_msg_digest = self.accumulate_public_expressions(
            &message_chunk_challenges,
            &[clk.expr(), m.get_subarray(0..16).expr()],
        );
        self.input_to_bus(bus_channel_idx, clk_msg_digest);

        /*
        for i in 0..8 {
            self.set_to_expression_first_row(
                &h_output.get(i),
                ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
            );
        }
        */

        // Set h_input to the initial hash if we are at the first block.
        // Otherwise set it to h_output
        for ((h_in, init), h_out) in h_input.iter().zip(initial_hash.iter()).zip(h_output.iter()) {
            self.set_to_expression(
                &h_in,
                first_block_bit.expr() * init.expr() + first_block_bit.not_expr() * h_out.expr(),
            );
        }

        self.blake2b_compress(
            clk,
            &m,
            &h_input,
            &h_output,
            &initial_hash_compress,
            &inversion_const,
            &t,
            &last_block_bit,
            &cycle_12_start_bit,
            &cycle_12_end_bit,
            operations,
        );

        let clk_hash_next = self.accumulate_expressions(
            &state_challenges,
            &[clk.expr(), h_output.get_subarray(0..8).expr()],
        );
        self.input_to_bus_filtered(bus_channel_idx, clk_hash_next, cycle_12_end_bit.expr());

        BLAKE2BGadget {
            t,
            m,
            first_block_bit,
            last_block_bit,
            cycle_12_start_bit,
            cycle_12_end_bit,
            msg_chunks,
            initial_hash,
            initial_hash_compress,
            hash_state,
            inversion_const,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn blake2b_compress(
        &mut self,
        clk: &ElementRegister,
        m: &ArrayRegister<U64Register>,
        h_input: &ArrayRegister<U64Register>,
        h_output: &ArrayRegister<U64Register>,
        iv_pub: &ArrayRegister<U64Register>,
        inversion_const_pub: &U64Register,
        t: &U64Register, // assumes t is not more than u64
        last_block_bit: &BitRegister,
        cycle_12_start_bit: &BitRegister,
        cycle_12_end_bit: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: U32Instructions,
    {
        // Need to create non public registers for IV and inversion_const.  Operating on public and private registers causes issues.
        let iv = self.alloc_array::<U64Register>(8);
        for i in 0..8 {
            self.set_to_expression(&iv.get(i), iv_pub.get(i).expr());
        }

        let inversion_const = self.alloc::<U64Register>();
        self.set_to_expression(&inversion_const, inversion_const_pub.expr());

        // Allocate the work vector
        // This is used to store the current value of the work vector during the mix loop
        let mut v_0 = self.alloc::<U64Register>();
        let mut v_1 = self.alloc::<U64Register>();
        let mut v_2 = self.alloc::<U64Register>();
        let mut v_3 = self.alloc::<U64Register>();
        let mut v_4 = self.alloc::<U64Register>();
        let mut v_5 = self.alloc::<U64Register>();
        let mut v_6 = self.alloc::<U64Register>();
        let mut v_7 = self.alloc::<U64Register>();
        let mut v_8 = self.alloc::<U64Register>();
        let mut v_9 = self.alloc::<U64Register>();
        let mut v_10 = self.alloc::<U64Register>();
        let mut v_11 = self.alloc::<U64Register>();
        let mut v_12 = self.alloc::<U64Register>();
        let mut v_13 = self.alloc::<U64Register>();
        let mut v_14 = self.alloc::<U64Register>();
        let mut v_15 = self.alloc::<U64Register>();

        // This is used to store the output work vector after mix
        let v_0_out = self.alloc::<U64Register>();
        let v_1_out = self.alloc::<U64Register>();
        let v_2_out = self.alloc::<U64Register>();
        let v_3_out = self.alloc::<U64Register>();
        let v_4_out = self.alloc::<U64Register>();
        let v_5_out = self.alloc::<U64Register>();
        let v_6_out = self.alloc::<U64Register>();
        let v_7_out = self.alloc::<U64Register>();
        let v_8_out = self.alloc::<U64Register>();
        let v_9_out = self.alloc::<U64Register>();
        let v_10_out = self.alloc::<U64Register>();
        let v_11_out = self.alloc::<U64Register>();
        let v_12_out = self.alloc::<U64Register>();
        let v_13_out = self.alloc::<U64Register>();
        let v_14_out = self.alloc::<U64Register>();
        let v_15_out = self.alloc::<U64Register>();

        /*
        self.set_to_expression_first_row(
            &v_0_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_1_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_2_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_3_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_4_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_5_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_6_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_7_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_8_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_9_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_10_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_11_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_12_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_13_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_14_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        self.set_to_expression_first_row(
            &v_15_out,
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
        );
        */

        // If this is the first bit of the 12 round mix cycle, initialize the work vector, else set it to v_out
        v_0 = self.select(cycle_12_start_bit, &v_0, &v_0_out);
        v_1 = self.select(cycle_12_start_bit, &v_1, &v_1_out);
        v_2 = self.select(cycle_12_start_bit, &v_2, &v_2_out);
        v_3 = self.select(cycle_12_start_bit, &v_3, &v_3_out);
        v_4 = self.select(cycle_12_start_bit, &v_4, &v_4_out);
        v_5 = self.select(cycle_12_start_bit, &v_5, &v_5_out);
        v_6 = self.select(cycle_12_start_bit, &v_6, &v_6_out);
        v_7 = self.select(cycle_12_start_bit, &v_7, &v_7_out);
        v_8 = self.select(cycle_12_start_bit, &v_8, &v_8_out);
        v_9 = self.select(cycle_12_start_bit, &v_9, &v_9_out);
        v_10 = self.select(cycle_12_start_bit, &v_10, &v_10_out);
        v_11 = self.select(cycle_12_start_bit, &v_11, &v_11_out);
        v_12 = self.select(cycle_12_start_bit, &v_12, &v_12_out);
        v_13 = self.select(cycle_12_start_bit, &v_13, &v_13_out);
        v_14 = self.select(cycle_12_start_bit, &v_14, &v_14_out);
        v_15 = self.select(cycle_12_start_bit, &v_15, &v_15_out);

        let v_12_for_start_bit = self.bitwise_xor(&v_12, t, operations);
        v_12 = self.select(cycle_12_start_bit, &v_12_for_start_bit, &v_12);

        let mut v_14_for_last_block = self.bitwise_xor(&v_14, &inversion_const, operations);
        v_14_for_last_block = self.select(cycle_12_start_bit, &v_14_for_last_block, &v_14);
        v_14 = self.select(last_block_bit, &v_14_for_last_block, &v_14);
        // Invert v[14] bits if this is the last block and we are at the start of the mix 12 cycle.

        self.blake2b_mix(
            &mut v_0,
            &mut v_4,
            &mut v_8,
            &mut v_12,
            &m.get(0),
            &m.get(1),
            operations,
        );

        self.blake2b_mix(
            &mut v_1,
            &mut v_5,
            &mut v_9,
            &mut v_13,
            &m.get(2),
            &m.get(3),
            operations,
        );

        self.blake2b_mix(
            &mut v_2,
            &mut v_6,
            &mut v_10,
            &mut v_14,
            &m.get(4),
            &m.get(5),
            operations,
        );

        self.blake2b_mix(
            &mut v_3,
            &mut v_7,
            &mut v_11,
            &mut v_15,
            &m.get(6),
            &m.get(7),
            operations,
        );

        self.blake2b_mix(
            &mut v_0,
            &mut v_5,
            &mut v_10,
            &mut v_15,
            &m.get(8),
            &m.get(9),
            operations,
        );

        self.blake2b_mix(
            &mut v_1,
            &mut v_6,
            &mut v_11,
            &mut v_12,
            &m.get(10),
            &m.get(11),
            operations,
        );

        self.blake2b_mix(
            &mut v_2,
            &mut v_7,
            &mut v_8,
            &mut v_13,
            &m.get(12),
            &m.get(13),
            operations,
        );

        self.blake2b_mix(
            &mut v_3,
            &mut v_4,
            &mut v_9,
            &mut v_14,
            &m.get(14),
            &m.get(15),
            operations,
        );

        self.set_to_expression_transition(&v_0_out, v_0.expr());
        self.set_to_expression_transition(&v_1_out, v_1.expr());
        self.set_to_expression_transition(&v_2_out, v_2.expr());
        self.set_to_expression_transition(&v_3_out, v_3.expr());
        self.set_to_expression_transition(&v_4_out, v_4.expr());
        self.set_to_expression_transition(&v_5_out, v_5.expr());
        self.set_to_expression_transition(&v_6_out, v_6.expr());
        self.set_to_expression_transition(&v_7_out, v_7.expr());
        self.set_to_expression_transition(&v_8_out, v_8.expr());
        self.set_to_expression_transition(&v_9_out, v_9.expr());
        self.set_to_expression_transition(&v_10_out, v_10.expr());
        self.set_to_expression_transition(&v_11_out, v_11.expr());
        self.set_to_expression_transition(&v_12_out, v_12.expr());
        self.set_to_expression_transition(&v_13_out, v_13.expr());
        self.set_to_expression_transition(&v_14_out, v_14.expr());
        self.set_to_expression_transition(&v_15_out, v_15.expr());

        let mut h_0_tmp = self.bitwise_xor(&h_input.get(0), &v_0_out, operations);
        let mut h_1_tmp = self.bitwise_xor(&h_input.get(1), &v_1_out, operations);
        let mut h_2_tmp = self.bitwise_xor(&h_input.get(2), &v_2_out, operations);
        let mut h_3_tmp = self.bitwise_xor(&h_input.get(3), &v_3_out, operations);
        let mut h_4_tmp = self.bitwise_xor(&h_input.get(4), &v_4_out, operations);
        let mut h_5_tmp = self.bitwise_xor(&h_input.get(5), &v_5_out, operations);
        let mut h_6_tmp = self.bitwise_xor(&h_input.get(6), &v_6_out, operations);
        let mut h_7_tmp = self.bitwise_xor(&h_input.get(7), &v_7_out, operations);

        h_0_tmp = self.bitwise_xor(&h_0_tmp, &v_8_out, operations);
        h_1_tmp = self.bitwise_xor(&h_1_tmp, &v_9_out, operations);
        h_2_tmp = self.bitwise_xor(&h_2_tmp, &v_10_out, operations);
        h_3_tmp = self.bitwise_xor(&h_3_tmp, &v_11_out, operations);
        h_4_tmp = self.bitwise_xor(&h_4_tmp, &v_12_out, operations);
        h_5_tmp = self.bitwise_xor(&h_5_tmp, &v_13_out, operations);
        h_6_tmp = self.bitwise_xor(&h_6_tmp, &v_14_out, operations);
        h_7_tmp = self.bitwise_xor(&h_7_tmp, &v_15_out, operations);

        self.set_to_expression_transition(
            &h_output.get(0).next(),
            cycle_12_end_bit.expr() * h_0_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(0).expr(),
        );
        self.set_to_expression_transition(
            &h_output.get(1).next(),
            cycle_12_end_bit.expr() * h_1_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(1).expr(),
        );
        self.set_to_expression_transition(
            &h_output.get(2).next(),
            cycle_12_end_bit.expr() * h_2_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(2).expr(),
        );
        self.set_to_expression_transition(
            &h_output.get(3).next(),
            cycle_12_end_bit.expr() * h_3_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(3).expr(),
        );
        self.set_to_expression_transition(
            &h_output.get(4).next(),
            cycle_12_end_bit.expr() * h_4_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(4).expr(),
        );
        self.set_to_expression_transition(
            &h_output.get(5).next(),
            cycle_12_end_bit.expr() * h_5_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(5).expr(),
        );
        self.set_to_expression_transition(
            &h_output.get(6).next(),
            cycle_12_end_bit.expr() * h_6_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(6).expr(),
        );
        self.set_to_expression_transition(
            &h_output.get(7).next(),
            cycle_12_end_bit.expr() * h_7_tmp.expr()
                + cycle_12_end_bit.not_expr() * h_input.get(7).expr(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn blake2b_mix(
        &mut self,
        v_a: &mut U64Register,
        v_b: &mut U64Register,
        v_c: &mut U64Register,
        v_d: &mut U64Register,
        x: &U64Register,
        y: &U64Register,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: U32Instructions,
    {
        *v_a = self.add_u64(v_a, v_b, operations);
        *v_a = self.add_u64(v_a, x, operations);

        *v_d = self.bitwise_xor(v_d, v_a, operations);
        *v_d = self.bit_rotate_right(v_d, 32, operations);

        *v_c = self.add_u64(v_c, v_d, operations);

        *v_b = self.bitwise_xor(v_b, v_c, operations);
        *v_b = self.bit_rotate_right(v_b, 24, operations);

        *v_a = self.add_u64(v_a, v_b, operations);
        *v_a = self.add_u64(v_a, y, operations);

        *v_d = self.bitwise_xor(v_d, v_a, operations);
        *v_d = self.bit_rotate_right(v_d, 16, operations);

        *v_c = self.add_u64(v_c, v_d, operations);

        *v_b = self.bitwise_xor(v_b, v_c, operations);
        *v_b = self.bit_rotate_right(v_b, 63, operations);
    }
}

impl BLAKE2BGadget {
    // TODO:  Need to revisit this
    pub fn write<F: Field, I: IntoIterator>(
        &self,
        padded_messages: I,
        message_lens: &[u64],
        writer: &TraceWriter<F>,
    ) where
        I::Item: Borrow<[u8]>,
    {
        let mut first_block_bit_values = Vec::new();
        let mut last_block_bit_values = Vec::new();
        let mut cycle_12_start_bit_values = Vec::new();
        let mut cycle_12_end_bit_values = Vec::new();
        let mut m_chunks = Vec::new();
        let mut t_values = Vec::new();

        // Public values
        let mut hash_values = Vec::new();
        let mut msg_chunks = Vec::<[F; 8]>::new();

        for (padded_msg, message_len) in padded_messages.into_iter().zip(message_lens.iter()) {
            let padded_msg = padded_msg.borrow();
            let num_chunks = padded_msg.len() / 128;

            first_block_bit_values.extend_from_slice(&[F::ONE; 12]);
            first_block_bit_values.extend_from_slice(&vec![F::ZERO; (num_chunks - 1) * 12]);
            last_block_bit_values.extend_from_slice(&vec![F::ZERO; num_chunks - 1]);
            last_block_bit_values.extend_from_slice(&[F::ONE; 12]);

            let mut state = INITIAL_HASH;
            let mut bytes_compressed = 0;
            for (chunk_num, chunk) in padded_msg.chunks_exact(128).enumerate() {
                cycle_12_start_bit_values.push(F::ONE);
                cycle_12_start_bit_values.extend_from_slice(&[F::ZERO; 11]);
                cycle_12_end_bit_values.extend_from_slice(&[F::ZERO; 11]);
                cycle_12_end_bit_values.push(F::ONE);

                let last_chunk = chunk_num == num_chunks - 1;

                if last_chunk {
                    bytes_compressed = *message_len;
                } else {
                    bytes_compressed += 128;
                }

                t_values.extend_from_slice(&[bytes_compressed; 12]);

                state = BLAKE2BGadget::compress(
                    chunk,
                    &mut state,
                    bytes_compressed,
                    chunk_num == num_chunks - 1,
                );
                hash_values.extend_from_slice(&state.map(u64_to_le_field_bytes::<F>));

                let chunk_array: [[F; 8]; 16] = chunk
                    .chunks_exact(8)
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|y| F::from_canonical_u8(*y))
                            .collect_vec()
                            .as_slice()
                            .try_into()
                            .expect("should be slice of 8 elements")
                    })
                    .collect_vec()
                    .as_slice()
                    .try_into()
                    .expect("should be slice of 16 elements");

                msg_chunks.extend_from_slice(&chunk_array);

                // Permute the messages
                for i in 0..12 {
                    let permuted_chunk_array = self.permute_msgs(&chunk_array, i);
                    m_chunks.push(permuted_chunk_array);
                }
            }
        }

        println!("hash_values.len() = {}", hash_values.len());
        assert!(
            hash_values.len() == 512 * 8,
            "Padded messages lengths do not add up"
        );

        // Write to the public registers
        writer.write_array(
            &self.initial_hash,
            INITIAL_HASH.map(u64_to_le_field_bytes),
            0,
        );

        writer.write_array(
            &self.initial_hash_compress,
            INITIAL_HASH_COMPRESS.map(u64_to_le_field_bytes),
            0,
        );
        writer.write_array(&self.hash_state, &hash_values, 0);
        writer.write_array(&self.msg_chunks, &msg_chunks, 0);
        writer.write(
            &self.inversion_const,
            &u64_to_le_field_bytes(INVERSION_CONST),
            0,
        );

        // Write to the the local registers
        for i in 0..512 * 12 {
            writer.write(&self.t, &u64_to_le_field_bytes(t_values[i]), i);
            writer.write(&self.first_block_bit, &first_block_bit_values[i], i);
            writer.write(&self.last_block_bit, &last_block_bit_values[i], i);
            writer.write(&self.cycle_12_start_bit, &cycle_12_start_bit_values[i], i);
            writer.write(&self.cycle_12_end_bit, &cycle_12_end_bit_values[i], i);
            writer.write_array(&self.m, &m_chunks[i], i);
        }
    }

    fn permute_msgs<T: Clone>(&self, arr: &[T], mix_round_num: usize) -> Vec<T> {
        assert!(mix_round_num <= 12);

        let permutation = SIGMA[mix_round_num % 10];
        let mut result = vec![arr[0].clone(); arr.len()];

        for (from_index, &to_index) in permutation.iter().enumerate() {
            result[to_index] = arr[from_index].clone();
        }

        result
    }

    pub fn compress(
        msg_chunk: &[u8],
        state: &mut [u64; 8],
        bytes_compressed: u64,
        last_chunk: bool,
    ) -> [u64; 8] {
        // Set up the work vector V
        let mut v: [u64; 16] = [0; 16];

        v[..8].copy_from_slice(&state[..8]);
        v[8..16].copy_from_slice(&INITIAL_HASH_COMPRESS);

        v[12] ^= bytes_compressed;
        if last_chunk {
            v[14] ^= INVERSION_CONST;
        }

        let msg_u64_chunks = msg_chunk
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect_vec();

        for i in 0..12 {
            let s = SIGMA[i % 10];

            BLAKE2BGadget::mix(
                &mut v,
                0,
                4,
                8,
                12,
                msg_u64_chunks[s[0]],
                msg_u64_chunks[s[1]],
            );
            BLAKE2BGadget::mix(
                &mut v,
                1,
                5,
                9,
                13,
                msg_u64_chunks[s[2]],
                msg_u64_chunks[s[3]],
            );
            BLAKE2BGadget::mix(
                &mut v,
                2,
                6,
                10,
                14,
                msg_u64_chunks[s[4]],
                msg_u64_chunks[s[5]],
            );
            BLAKE2BGadget::mix(
                &mut v,
                3,
                7,
                11,
                15,
                msg_u64_chunks[s[6]],
                msg_u64_chunks[s[7]],
            );

            BLAKE2BGadget::mix(
                &mut v,
                0,
                5,
                10,
                15,
                msg_u64_chunks[s[8]],
                msg_u64_chunks[s[9]],
            );
            BLAKE2BGadget::mix(
                &mut v,
                1,
                6,
                11,
                12,
                msg_u64_chunks[s[10]],
                msg_u64_chunks[s[11]],
            );
            BLAKE2BGadget::mix(
                &mut v,
                2,
                7,
                8,
                13,
                msg_u64_chunks[s[12]],
                msg_u64_chunks[s[13]],
            );
            BLAKE2BGadget::mix(
                &mut v,
                3,
                4,
                9,
                14,
                msg_u64_chunks[s[14]],
                msg_u64_chunks[s[15]],
            );
        }

        for i in 0..8 {
            state[i] ^= v[i];
        }

        for i in 0..8 {
            state[i] ^= v[i + 8];
        }

        *state
    }

    fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }

    pub fn pad(msg: &[u8]) -> Vec<u8> {
        if (msg.len() % 128 == 0) && (!msg.is_empty()) {
            msg.to_vec()
        } else {
            let padlen = 128 - (msg.len() % 128);

            let mut padded_msg = Vec::new();
            padded_msg.extend_from_slice(msg);
            padded_msg.extend_from_slice(&vec![0u8; padlen]);
            padded_msg
        }
    }
}

#[cfg(test)]
mod tests {

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::PrimeField64;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Copy)]
    pub struct BLAKE2BTest;

    impl AirParameters for BLAKE2BTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_FREE_COLUMNS: usize = 2498;
        const EXTENDED_COLUMNS: usize = 4737;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_blake2b_stark() {
        type F = GoldilocksField;
        type L = BLAKE2BTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Blake2b test", log::Level::Debug);

        let mut builder = AirBuilder::<L>::new();
        let clk = builder.clock();

        let (mut operations, table) = builder.byte_operations();

        let mut bus = builder.new_bus();
        let channel_idx = bus.new_channel(&mut builder);

        let blake_gadget = builder.process_blake2b(&clk, &mut bus, channel_idx, &mut operations);

        builder.register_byte_lookup(operations, &table);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);
        let writer = generator.new_writer();

        let msg = b"";
        let digest = "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8";

        let mut padded_messages = Vec::new();
        let mut msg_lens = Vec::new();

        for i in 0..512 {
            padded_messages.push(BLAKE2BGadget::pad(msg).into_iter().collect::<Vec<_>>());
            msg_lens.push(msg.len() as u64);
        }

        timed!(timing, "Write the execusion trace", {
            table.write_table_entries(&writer);
            blake_gadget.write(padded_messages, msg_lens.as_slice(), &writer);
            for i in 0..L::num_rows() {
                writer.write_row_instructions(&generator.air_data, i);
                let last_block_bit = writer.read(&blake_gadget.last_block_bit, i);
                let cycle_12_end_bit = writer.read(&blake_gadget.cycle_12_end_bit, i);
                if last_block_bit == F::ONE && cycle_12_end_bit == F::ONE {
                    let hash: [[GoldilocksField; 8]; 4] =
                        writer.read_array(&blake_gadget.hash_state.get_subarray(0..8), i);
                    let calculated_hash_bytes = hash
                        .iter()
                        .flatten()
                        .map(|x| x.to_canonical_u64() as u8)
                        .collect_vec();
                    //assert_eq!(calculated_hash_bytes, expected_digest);
                }
            }
            table.write_multiplicities(&writer);
        });

        let public_inputs = writer.0.public.read().unwrap().clone();
        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        timed!(
            timing,
            "Stark proof and verify",
            test_starky(&stark, &config, &generator, &public_inputs)
        );

        // // Generate recursive proof
        // timed!(
        //     timing,
        //     "Recursive proof generation and verification",
        //     test_recursive_starky(stark, config, generator, &public_inputs)
        // );

        timing.print();
    }
}
