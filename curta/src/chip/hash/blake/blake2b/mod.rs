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

const NUM_MIX_ROUNDS: usize = 12;
const MSG_ARRAY_SIZE: usize = 16;
const HASH_ARRAY_SIZE: usize = 8;
const CYCLE_12: usize = 12;

#[derive(Debug, Clone)]
pub struct BLAKE2BGadget {
    pub padding_bit: BitRegister,
    pub cycle_12_start_bit: BitRegister,
    pub cycle_12_end_bit: BitRegister,

    pub m: ArrayRegister<U64Register>,
    pub t: U64Register,
    pub last_chunk_bit: BitRegister,
    pub h_input: ArrayRegister<U64Register>,
    pub h_output: ArrayRegister<U64Register>,
    pub unused_row: BitRegister,

    // Public values
    pub initial_hash: ArrayRegister<U64Register>,
    pub initial_hash_compress: ArrayRegister<U64Register>,
    pub inversion_const: U64Register,
    pub msg_chunks: ArrayRegister<U64Register>,
    pub t_public: ArrayRegister<U64Register>,
    pub last_chunk_bit_public: ArrayRegister<BitRegister>,
    pub hash_state: ArrayRegister<U64Register>,
}

#[derive(Debug, Clone)]
pub struct BLAKE2BPublicData<T> {
    pub hash_state: Vec<U64Value<T>>,
    pub end_bits: Vec<T>,
}

const INITIAL_HASH: [u64; HASH_ARRAY_SIZE] = [
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
const INITIAL_HASH_COMPRESS: [u64; HASH_ARRAY_SIZE] = [
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
const SIGMA: [[usize; MSG_ARRAY_SIZE]; SIGMA_LEN] = [
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
        let num_chunks = L::num_rows() / NUM_MIX_ROUNDS;

        // Registers to be written to
        let m = self.alloc_array::<U64Register>(MSG_ARRAY_SIZE);
        let t = self.alloc::<U64Register>();
        let last_chunk_bit = self.alloc::<BitRegister>();
        let h_input = self.alloc_array::<U64Register>(HASH_ARRAY_SIZE);
        let h_output = self.alloc_array::<U64Register>(HASH_ARRAY_SIZE);
        let unused_row = self.alloc::<BitRegister>();
        let padding_bit = self.alloc::<BitRegister>();

        // Public values
        let initial_hash = self.alloc_array_public::<U64Register>(HASH_ARRAY_SIZE);
        let initial_hash_compress = self.alloc_array_public::<U64Register>(HASH_ARRAY_SIZE);
        let inversion_const = self.alloc_public::<U64Register>();
        let msg_chunks = self.alloc_array_public::<U64Register>(num_chunks * MSG_ARRAY_SIZE);
        let t_public = self.alloc_array_public::<U64Register>(num_chunks);
        let last_chunk_bit_public = self.alloc_array_public::<BitRegister>(num_chunks);
        let hash_state = self.alloc_array_public::<U64Register>(num_chunks * HASH_ARRAY_SIZE);

        let (cycle_12_start_bit, cycle_12_end_bit) = self.cycle_12();

        // Set h_input to the initial hash if we are at the first block and at the first loop of the cycle_12
        // Otherwise set it to h_output
        for (h_in, init) in h_input.iter().zip(initial_hash.iter()) {
            self.set_to_expression_first_row(&h_in, init.expr());
        }

        self.blake2b_compress(
            &m,
            &h_input,
            &h_output,
            &initial_hash_compress,
            &inversion_const,
            &t,
            &last_chunk_bit,
            &cycle_12_start_bit,
            &cycle_12_end_bit,
            &unused_row,
            operations,
        );

        for ((h_in, init), h_out) in h_input.iter().zip(initial_hash.iter()).zip(h_output.iter()) {
            self.set_to_expression_transition(
                &h_in.next(),
                last_chunk_bit.expr()
                    * (cycle_12_end_bit.expr() * init.expr()
                        + cycle_12_end_bit.not_expr() * h_out.expr())
                    + (last_chunk_bit.not_expr() * h_out.expr()),
            );
        }

        self.add_bus_constraints(
            clk,
            bus,
            bus_channel_idx,
            num_chunks,
            &cycle_12_start_bit,
            &cycle_12_end_bit,
            &m,
            &t,
            &last_chunk_bit,
            &h_output,
            &padding_bit,
            &msg_chunks,
            &t_public,
            &last_chunk_bit_public,
            &hash_state,
        );

        BLAKE2BGadget {
            padding_bit,
            cycle_12_start_bit,
            cycle_12_end_bit,

            m,
            t,
            last_chunk_bit,
            h_input,
            h_output,
            unused_row,

            initial_hash,
            initial_hash_compress,
            inversion_const,
            msg_chunks,
            t_public,
            last_chunk_bit_public,
            hash_state,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_bus_constraints(
        &mut self,
        clk: &ElementRegister,
        bus: &mut Bus<L::CubicParams>,
        bus_channel_idx: usize,
        num_chunks: usize,
        cycle_12_start_bit: &BitRegister,
        cycle_12_end_bit: &BitRegister,

        m: &ArrayRegister<U64Register>,
        t: &U64Register,
        last_chunk_bit: &BitRegister,
        h_output: &ArrayRegister<U64Register>,
        padding_bit: &BitRegister,

        msg_chunks: &ArrayRegister<U64Register>,
        t_public: &ArrayRegister<U64Register>,
        last_chunk_bit_public: &ArrayRegister<BitRegister>,
        hash_state: &ArrayRegister<U64Register>,
    ) where
        L::Instruction: U32Instructions,
    {
        // Get message chunk challenges
        let message_chunk_challenges = self
            .alloc_challenge_array::<CubicRegister>(U64Register::size_of() * MSG_ARRAY_SIZE + 1);

        // Get hash state challenges
        let state_challenges = self
            .alloc_challenge_array::<CubicRegister>(U64Register::size_of() * HASH_ARRAY_SIZE + 1);

        // Get the last chunk callenges
        let last_chunk_challenges = self.alloc_challenge_array::<CubicRegister>(2);

        // Get the t challenges
        let t_challenges = self.alloc_challenge_array::<CubicRegister>(U64Register::size_of() + 1);

        // Put public hash state, end_bits, and all the msg chunk permutations into the bus
        for i in 0..num_chunks {
            for j in 0..NUM_MIX_ROUNDS {
                let row = i * NUM_MIX_ROUNDS + j;
                let row_expr =
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(row));

                // For the first row of every cycle send the public t to the bus
                if j == 0 {
                    let t_digest = self.accumulate_public_expressions(
                        &t_challenges,
                        &[row_expr.clone(), t_public.get(i).expr()],
                    );
                    bus.output_global_value(&t_digest);
                }

                // For the last row of every cycle, send the public hash state to the bus
                if j == 11 {
                    let state_digest = self.accumulate_public_expressions(
                        &state_challenges,
                        &[
                            row_expr.clone(),
                            hash_state
                                .get_subarray(
                                    i * HASH_ARRAY_SIZE..i * HASH_ARRAY_SIZE + HASH_ARRAY_SIZE,
                                )
                                .expr(),
                        ],
                    );

                    bus.output_global_value(&state_digest);
                }

                // For every row of the cycle, send the last chunk bit to the bus
                let last_chunk_digest = self.accumulate_public_expressions(
                    &last_chunk_challenges,
                    &[row_expr.clone(), last_chunk_bit_public.get(i).expr()],
                );

                bus.output_global_value(&last_chunk_digest);

                // For every row of the cycle, send the message chunk permutation to the bus
                let sigma = SIGMA[j % SIGMA_LEN];
                let mut values = Vec::new();

                values.push(row_expr.clone());

                for index in sigma.iter() {
                    values.push(msg_chunks.get(i * MSG_ARRAY_SIZE + index).expr());
                }

                let msg_digest = self
                    .accumulate_public_expressions(&message_chunk_challenges, values.as_slice());

                bus.output_global_value(&msg_digest);
            }
        }

        let t_digest = self.accumulate_expressions(&t_challenges, &[clk.expr(), t.expr()]);
        self.input_to_bus_filtered(
            bus_channel_idx,
            t_digest,
            cycle_12_start_bit.expr() * padding_bit.not_expr(),
        );
        self.assert_expression_zero(
            t.expr() * cycle_12_start_bit.not_expr() * padding_bit.not_expr(),
        );

        let clk_hash_next = self.accumulate_expressions(
            &state_challenges,
            &[clk.expr(), h_output.get_subarray(0..HASH_ARRAY_SIZE).expr()],
        );
        self.input_to_bus_filtered(
            bus_channel_idx,
            clk_hash_next,
            cycle_12_end_bit.expr() * padding_bit.not_expr(),
        );

        let clk_last_chunk_digest = self
            .accumulate_expressions(&last_chunk_challenges, &[clk.expr(), last_chunk_bit.expr()]);
        self.input_to_bus_filtered(
            bus_channel_idx,
            clk_last_chunk_digest,
            padding_bit.not_expr(),
        );

        let clk_msg_digest = self.accumulate_expressions(
            &message_chunk_challenges,
            &[clk.expr(), m.get_subarray(0..MSG_ARRAY_SIZE).expr()],
        );
        self.input_to_bus_filtered(bus_channel_idx, clk_msg_digest, padding_bit.not_expr());
    }

    fn cycle_12(&mut self) -> (BitRegister, BitRegister) {
        let cycle_12_registers = self.alloc_array::<BitRegister>(CYCLE_12);

        // Set the cycle 12 registers first row
        self.set_to_expression_first_row(
            &cycle_12_registers.get(0),
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(1)),
        );

        for i in 1..CYCLE_12 {
            self.set_to_expression_first_row(
                &cycle_12_registers.get(i),
                ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
            );
        }

        // Set transition constraint for the cycle_12_registers
        for i in 0..CYCLE_12 {
            let next_i = (i + 1) % CYCLE_12;

            self.set_to_expression_transition(
                &cycle_12_registers.get(next_i).next(),
                cycle_12_registers.get(i).expr(),
            );
        }

        (
            cycle_12_registers.get(0),
            cycle_12_registers.get(CYCLE_12 - 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn blake2b_compress(
        &mut self,
        m: &ArrayRegister<U64Register>,
        h_input: &ArrayRegister<U64Register>,
        h_output: &ArrayRegister<U64Register>,
        iv_pub: &ArrayRegister<U64Register>,
        inversion_const_pub: &U64Register,
        t: &U64Register, // assumes t is not more than u64
        last_chunk_bit: &BitRegister,
        cycle_12_start_bit: &BitRegister,
        cycle_12_end_bit: &BitRegister,
        unused_row: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: U32Instructions,
    {
        // Need to create non public registers for IV and inversion_const.  Operating on public and private registers causes issues.
        let iv = self.alloc_array::<U64Register>(HASH_ARRAY_SIZE);
        for i in 0..HASH_ARRAY_SIZE {
            self.set_to_expression(&iv.get(i), iv_pub.get(i).expr());
        }

        let inversion_const = self.alloc::<U64Register>();
        self.set_to_expression(&inversion_const, inversion_const_pub.expr());

        // Allocate the work vector
        // This is used to store the current value of the work vector during the mix loop
        let v_0_orig = self.alloc::<U64Register>();
        let v_1_orig = self.alloc::<U64Register>();
        let v_2_orig = self.alloc::<U64Register>();
        let v_3_orig = self.alloc::<U64Register>();
        let v_4_orig = self.alloc::<U64Register>();
        let v_5_orig = self.alloc::<U64Register>();
        let v_6_orig = self.alloc::<U64Register>();
        let v_7_orig = self.alloc::<U64Register>();
        let v_8_orig = self.alloc::<U64Register>();
        let v_9_orig = self.alloc::<U64Register>();
        let v_10_orig = self.alloc::<U64Register>();
        let v_11_orig = self.alloc::<U64Register>();
        let v_12_orig = self.alloc::<U64Register>();
        let v_13_orig = self.alloc::<U64Register>();
        let v_14_orig = self.alloc::<U64Register>();
        let v_15_orig = self.alloc::<U64Register>();

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

        // If this is the first bit of the 12 round mix cycle, initialize the work vector, else set it to v_out
        let mut v_0 = self.select(cycle_12_start_bit, &h_input.get(0), &v_0_orig);
        let mut v_1 = self.select(cycle_12_start_bit, &h_input.get(1), &v_1_orig);
        let mut v_2 = self.select(cycle_12_start_bit, &h_input.get(2), &v_2_orig);
        let mut v_3 = self.select(cycle_12_start_bit, &h_input.get(3), &v_3_orig);
        let mut v_4 = self.select(cycle_12_start_bit, &h_input.get(4), &v_4_orig);
        let mut v_5 = self.select(cycle_12_start_bit, &h_input.get(5), &v_5_orig);
        let mut v_6 = self.select(cycle_12_start_bit, &h_input.get(6), &v_6_orig);
        let mut v_7 = self.select(cycle_12_start_bit, &h_input.get(7), &v_7_orig);
        let mut v_8 = self.select(cycle_12_start_bit, &iv.get(0), &v_8_orig);
        let mut v_9 = self.select(cycle_12_start_bit, &iv.get(1), &v_9_orig);
        let mut v_10 = self.select(cycle_12_start_bit, &iv.get(2), &v_10_orig);
        let mut v_11 = self.select(cycle_12_start_bit, &iv.get(3), &v_11_orig);
        let mut v_12 = self.select(cycle_12_start_bit, &iv.get(4), &v_12_orig);
        let mut v_13 = self.select(cycle_12_start_bit, &iv.get(5), &v_13_orig);
        let mut v_14 = self.select(cycle_12_start_bit, &iv.get(6), &v_14_orig);
        let mut v_15 = self.select(cycle_12_start_bit, &iv.get(7), &v_15_orig);

        let v_12_for_start_bit = self.bitwise_xor(&v_12, t, operations);
        v_12 = self.select(cycle_12_start_bit, &v_12_for_start_bit, &v_12);

        let mut v_14_for_last_block = self.bitwise_xor(&v_14, &inversion_const, operations);
        v_14_for_last_block = self.select(cycle_12_start_bit, &v_14_for_last_block, &v_14);
        v_14 = self.select(last_chunk_bit, &v_14_for_last_block, &v_14);
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

        self.set_to_expression(&v_0_out, v_0.expr());
        self.set_to_expression(&v_1_out, v_1.expr());
        self.set_to_expression(&v_2_out, v_2.expr());
        self.set_to_expression(&v_3_out, v_3.expr());
        self.set_to_expression(&v_4_out, v_4.expr());
        self.set_to_expression(&v_5_out, v_5.expr());
        self.set_to_expression(&v_6_out, v_6.expr());
        self.set_to_expression(&v_7_out, v_7.expr());
        self.set_to_expression(&v_8_out, v_8.expr());
        self.set_to_expression(&v_9_out, v_9.expr());
        self.set_to_expression(&v_10_out, v_10.expr());
        self.set_to_expression(&v_11_out, v_11.expr());
        self.set_to_expression(&v_12_out, v_12.expr());
        self.set_to_expression(&v_13_out, v_13.expr());
        self.set_to_expression(&v_14_out, v_14.expr());
        self.set_to_expression(&v_15_out, v_15.expr());

        self.set_to_expression_transition(&v_0_orig.next(), v_0_out.expr());
        self.set_to_expression_transition(&v_1_orig.next(), v_1_out.expr());
        self.set_to_expression_transition(&v_2_orig.next(), v_2_out.expr());
        self.set_to_expression_transition(&v_3_orig.next(), v_3_out.expr());
        self.set_to_expression_transition(&v_4_orig.next(), v_4_out.expr());
        self.set_to_expression_transition(&v_5_orig.next(), v_5_out.expr());
        self.set_to_expression_transition(&v_6_orig.next(), v_6_out.expr());
        self.set_to_expression_transition(&v_7_orig.next(), v_7_out.expr());
        self.set_to_expression_transition(&v_8_orig.next(), v_8_out.expr());
        self.set_to_expression_transition(&v_9_orig.next(), v_9_out.expr());
        self.set_to_expression_transition(&v_10_orig.next(), v_10_out.expr());
        self.set_to_expression_transition(&v_11_orig.next(), v_11_out.expr());
        self.set_to_expression_transition(&v_12_orig.next(), v_12_out.expr());
        self.set_to_expression_transition(&v_13_orig.next(), v_13_out.expr());
        self.set_to_expression_transition(&v_14_orig.next(), v_14_out.expr());
        self.set_to_expression_transition(&v_15_orig.next(), v_15_out.expr());

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

        let u64_register_zero = ArithmeticExpression::from_constant_vec(vec![L::Field::ZERO; 8]);

        // set h_output to zero if within the padded section
        self.set_to_expression(
            &h_output.get(0),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_0_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(0).expr())),
        );
        self.set_to_expression(
            &h_output.get(1),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_1_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(1).expr())),
        );
        self.set_to_expression(
            &h_output.get(2),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_2_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(2).expr())),
        );
        self.set_to_expression(
            &h_output.get(3),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_3_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(3).expr())),
        );
        self.set_to_expression(
            &h_output.get(4),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_4_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(4).expr())),
        );
        self.set_to_expression(
            &h_output.get(5),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_5_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(5).expr())),
        );
        self.set_to_expression(
            &h_output.get(6),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_6_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(6).expr())),
        );
        self.set_to_expression(
            &h_output.get(7),
            unused_row.expr() * u64_register_zero
                + (unused_row.not_expr()
                    * (cycle_12_end_bit.expr() * h_7_tmp.expr()
                        + cycle_12_end_bit.not_expr() * h_input.get(7).expr())),
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
    pub fn write<F: Field, I: IntoIterator>(
        &self,
        padded_messages: I,
        message_lens: &[u64],
        writer: &TraceWriter<F>,
        num_rows: usize,
    ) where
        I::Item: Borrow<[u8]>,
    {
        let max_num_chunks = num_rows / NUM_MIX_ROUNDS;

        let mut last_chunk_bit_values = Vec::new();
        let mut cycle_12_start_bit_values = Vec::new();
        let mut cycle_12_end_bit_values = Vec::new();
        let mut m_chunks = Vec::new();
        let mut t_values = Vec::new();

        // Public values
        let mut hash_values = Vec::new();
        let mut msg_chunks = Vec::<[F; 8]>::new();
        let mut last_chunk_bit_public = Vec::new();
        let mut t_values_public = Vec::new();

        let mut num_written_chunks = 0usize;
        for (padded_msg, message_len) in padded_messages.into_iter().zip(message_lens.iter()) {
            let padded_msg = padded_msg.borrow();
            let msg_num_chunks = padded_msg.len() / 128;

            last_chunk_bit_values
                .extend_from_slice(&vec![F::ZERO; (msg_num_chunks - 1) * NUM_MIX_ROUNDS]);
            last_chunk_bit_values.extend_from_slice(&[F::ONE; NUM_MIX_ROUNDS]);

            let mut state = INITIAL_HASH;
            let mut bytes_compressed = 0;
            for (chunk_num, chunk) in padded_msg.chunks_exact(128).enumerate() {
                cycle_12_start_bit_values.push(F::ONE);
                cycle_12_start_bit_values.extend_from_slice(&[F::ZERO; NUM_MIX_ROUNDS - 1]);
                cycle_12_end_bit_values.extend_from_slice(&[F::ZERO; NUM_MIX_ROUNDS - 1]);
                cycle_12_end_bit_values.push(F::ONE);

                let last_chunk = chunk_num == msg_num_chunks - 1;

                if last_chunk {
                    bytes_compressed = *message_len;
                    last_chunk_bit_public.push(F::ONE);
                } else {
                    bytes_compressed += 128;
                    last_chunk_bit_public.push(F::ZERO);
                }

                t_values.push(bytes_compressed);
                t_values.extend_from_slice(&[0; NUM_MIX_ROUNDS - 1]);
                t_values_public.push(u64_to_le_field_bytes::<F>(bytes_compressed));

                state = BLAKE2BGadget::compress(
                    chunk,
                    &mut state,
                    bytes_compressed,
                    chunk_num == msg_num_chunks - 1,
                );

                hash_values.extend_from_slice(&state.map(u64_to_le_field_bytes::<F>));

                let chunk_array: [[F; 8]; MSG_ARRAY_SIZE] = chunk
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
                for i in 0..NUM_MIX_ROUNDS {
                    let permuted_chunk_array = self.permute_msgs(&chunk_array, i);
                    m_chunks.push(permuted_chunk_array);
                }

                num_written_chunks += 1;
                assert!(num_written_chunks <= max_num_chunks, "Too many chunks");
            }
        }

        assert!(hash_values.len() == num_written_chunks * 8);

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

        writer.write(
            &self.inversion_const,
            &u64_to_le_field_bytes(INVERSION_CONST),
            0,
        );

        writer.write_array(&self.hash_state, &hash_values, 0);

        writer.write_array(&self.last_chunk_bit_public, &last_chunk_bit_public, 0);

        writer.write_array(&self.msg_chunks, &msg_chunks, 0);

        writer.write_array(&self.t_public, &t_values_public, 0);

        // Write to the the local registers
        for i in 0..num_written_chunks * NUM_MIX_ROUNDS {
            writer.write(&self.t, &u64_to_le_field_bytes(t_values[i]), i);
            writer.write(&self.last_chunk_bit, &last_chunk_bit_values[i], i);
            writer.write_array(&self.m, &m_chunks[i], i);
            writer.write(&self.unused_row, &F::ZERO, i);
            writer.write(&self.padding_bit, &F::ZERO, i);
        }

        let num_padding_bits = num_rows % NUM_MIX_ROUNDS;

        // Need to pad the rest of the rows
        //for i in num_rows_to_write..num_rows {
        for i in num_written_chunks * NUM_MIX_ROUNDS..num_rows {
            writer.write(&self.unused_row, &F::ONE, i);

            if i >= num_rows - num_padding_bits {
                writer.write(&self.padding_bit, &F::ONE, i);
            } else {
                writer.write(&self.padding_bit, &F::ZERO, i);
            }
        }
    }

    fn permute_msgs<T: Clone>(&self, arr: &[T], mix_round_num: usize) -> Vec<T> {
        assert!(mix_round_num <= NUM_MIX_ROUNDS);

        let permutation = SIGMA[mix_round_num % 10];
        let mut result = vec![arr[0].clone(); arr.len()];

        for (to_index, &from_index) in permutation.iter().enumerate() {
            result[to_index] = arr[from_index].clone();
        }

        result
    }

    pub fn compress(
        msg_chunk: &[u8],
        state: &mut [u64; HASH_ARRAY_SIZE],
        bytes_compressed: u64,
        last_chunk: bool,
    ) -> [u64; 8] {
        // Set up the work vector V
        let mut v: [u64; 16] = [0; 16];

        v[..8].copy_from_slice(&state[..HASH_ARRAY_SIZE]);
        v[8..16].copy_from_slice(&INITIAL_HASH_COMPRESS);

        v[12] ^= bytes_compressed;
        if last_chunk {
            v[14] ^= INVERSION_CONST;
        }

        let msg_u64_chunks = msg_chunk
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect_vec();

        for i in 0..NUM_MIX_ROUNDS {
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

        const NUM_FREE_COLUMNS: usize = 2509;
        const EXTENDED_COLUMNS: usize = 4755;
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
        builder.constrain_bus(bus);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);
        let writer = generator.new_writer();

        let msgs = [
            // 1 block
            hex::decode("").unwrap(),

            // 1 block
            hex::decode("092005a6f7a58a98df5f9b8d186b9877f12b603aa06c7debf0f610d5a49f9ed7262b5e095b309af2b0eae1c554e03b6cc4a5a0df207b662b329623f27fdce8d088554d82b1e63bedeb3fe9bd7754c7deccdfe277bcbfad4bbaff6302d3488bd2a8565f4f6e753fc7942fa29051e258da2e06d13b352220b9eadb31d8ead7f88b").unwrap(),

            // 8 blocks
            hex::decode("092005a6f7a58a98df5f9b8d186b9877f12b603aa06c7debf0f610d5a49f9ed7262b5e095b309af2b0eae1c554e03b6cc4a5a0df207b662b329623f27fdce8d088554d82b1e63bedeb3fe9bd7754c7deccdfe277bcbfad4bbaff6302d3488bd2a8565f4f6e753fc7942fa29051e258da2e06d13b352220b9eadb31d8ead7f88b244f13c0835db4a3909cee6106b276684aba0f8d8b1b0ba02dff4d659b081adfeab6f3a26d7fd65eff7c72a539dbeee68a9497476b69082958eae7d6a7f0f1d5a1b99a0a349691e80429667831f9b818431514bb2763e26e94a65428d22f3827d491c474c7a1885fe1d2d557e27bbcd81bffa9f3a507649e623b47681d6c9893301d8f635ec49e983cc537c4b81399bb24027ac4be709ce1a4eeb448e98a9aecfe249696419a67cb9e0f29d0297d840048bddf6612a383f37d7b96348a1bc5f1f9ac6eed6eb911dc43e120c8480e0258a6b33e0b91734cc64f144827053b17ae91c62e6866d8b68c1b0e53df0d0f0f4f187278db30c7b95d2741f4d0c8c59507984482b48d356ce8e299268b100c61a9ba5f96a757cf98150683a3e8aa85484a4590b293b6ec62c77f022542a73651a42b50f05a8d10bbb546746ca82221ca3b18105a05e4a7ea9c9d5096a37c8b3ce1a9c62ebd7badd7ee6f1c6e5961a08d066d5e025e08e3ec72531c476098287b13295fa606fab8275418e0c4c54f236c9e73fbfdaa00a5205310cb0d1bd54175647482fae300cc66b36e7846e82288e9f0290d9479d0c1998373900dfb72900d1c9f55c018dd7eeed4ce0e988bb3da03a22910ddec7c51b2eab4d96831a8b9e84a42cebdadae62bdea26ca7b0c640e8a21f86c72277ed20efe15bab1abcf34656e7d2336e42133fa99331e874b5458b28fabe6cb62c4606ee7046d07bc9e5eec2246068396590b59194c10bbe82f7c8b5ddea0d85a4cf74a91c85d7f90873bfbdc40c8c939377bec9a26d66b895a1bbeaa94028d6eafa1c0d6218077d174cc59cea6f2ea17ef1c002160e549f43b03112b0a978fd659c69448273e35554e21bac35458fe2b199f8b8fb81a6488ee99c734e2eefb4dd06c686ca29cdb2173a53ec8322a6cb9128e3b7cdf4bf5a5c2e8906b840bd86fa97ef694a34fd47740c2d44ff7378d773ee090903796a719697e67d8df4bc26d8aeb83ed380c04fe8aa4f23678989ebffd29c647eb96d4999b4a6736dd66c7a479fe0352fda60876f173519b4e567f0a0f0798d25e198603c1c5569b95fefa2edb64720ba97bd4d5f82614236b3a1f5deb344df02d095fccfe1db9b000f38ebe212f804ea0fbbeb645b8375e21d27f5381de0e0c0156f2fa3a0a0a055b8afe90b542f6e0fffb744f1dba74e34bb4d3ea6c84e49796f5e549781a2f5c2dc01d7b8e814661b5e2d2a51a258b2f7032a83082e6e36a5e51ef9af960b058").unwrap(),

            // 8 blocks
            hex::decode("092005a6f7a58a98df5f9b8d186b9877f12b603aa06c7debf0f610d5a49f9ed7262b5e095b309af2b0eae1c554e03b6cc4a5a0df207b662b329623f27fdce8d088554d82b1e63bedeb3fe9bd7754c7deccdfe277bcbfad4bbaff6302d3488bd2a8565f4f6e753fc7942fa29051e258da2e06d13b352220b9eadb31d8ead7f88b244f13c0835db4a3909cee6106b276684aba0f8d8b1b0ba02dff4d659b081adfeab6f3a26d7fd65eff7c72a539dbeee68a9497476b69082958eae7d6a7f0f1d5a1b99a0a349691e80429667831f9b818431514bb2763e26e94a65428d22f3827d491c474c7a1885fe1d2d557e27bbcd81bffa9f3a507649e623b47681d6c9893301d8f635ec49e983cc537c4b81399bb24027ac4be709ce1a4eeb448e98a9aecfe249696419a67cb9e0f29d0297d840048bddf6612a383f37d7b96348a1bc5f1f9ac6eed6eb911dc43e120c8480e0258a6b33e0b91734cc64f144827053b17ae91c62e6866d8b68c1b0e53df0d0f0f4f187278db30c7b95d2741f4d0c8c59507984482b48d356ce8e299268b100c61a9ba5f96a757cf98150683a3e8aa85484a4590b293b6ec62c77f022542a73651a42b50f05a8d10bbb546746ca82221ca3b18105a05e4a7ea9c9d5096a37c8b3ce1a9c62ebd7badd7ee6f1c6e5961a08d066d5e025e08e3ec72531c476098287b13295fa606fab8275418e0c4c54f236c9e73fbfdaa00a5205310cb0d1bd54175647482fae300cc66b36e7846e82288e9f0290d9479d0c1998373900dfb72900d1c9f55c018dd7eeed4ce0e988bb3da03a22910ddec7c51b2eab4d96831a8b9e84a42cebdadae62bdea26ca7b0c640e8a21f86c72277ed20efe15bab1abcf34656e7d2336e42133fa99331e874b5458b28fabe6cb62c4606ee7046d07bc9e5eec2246068396590b59194c10bbe82f7c8b5ddea0d85a4cf74a91c85d7f90873bfbdc40c8c939377bec9a26d66b895a1bbeaa94028d6eafa1c0d6218077d174cc59cea6f2ea17ef1c002160e549f43b03112b0a978fd659c69448273e35554e21bac35458fe2b199f8b8fb81a6488ee99c734e2eefb4dd06c686ca29cdb2173a53ec8322a6cb9128e3b7cdf4bf5a5c2e8906b840bd86fa97ef694a34fd47740c2d44ff7378d773ee090903796a719697e67d8df4bc26d8aeb83ed380c04fe8aa4f23678989ebffd29c647eb96d4999b4a6736dd66c7a479fe0352fda60876f173519b4e567f0a0f0798d25e198603c1c5569b95fefa2edb64720ba97bd4d5f82614236b3a1f5deb344df02d095fccfe1db9b000f38ebe212f804ea0fbbeb645b8375e21d27f5381de0e0c0156f2fa3a0a0a055b8afe90b542f6e0fffb744f1dba74e34bb4d3ea6c84e49796f5e549781a2f5c2dc01d7b8e814661b5e2d2a51a258b2f7032a83082e6e36a5e51").unwrap(),
        ];

        let digests = [
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8",
            "dad415aa819ebb585ce8ee1c1fa883804f405f6d8a6a0992628fb3bdaab5b42e",
            "022bfe46002fe82ab0c451574898fafaeb36283825aab39ddf825dc48a1c0970",
            "ad58001fb7de22fada15e574b17b2d2485183320cdc14f78625574328fedba84",
        ];

        let mut padded_messages = Vec::new();
        let mut msg_lens = Vec::new();

        for _i in 0..300 {
            for msg in msgs.iter() {
                padded_messages.push(BLAKE2BGadget::pad(msg).into_iter().collect::<Vec<_>>());
                msg_lens.push(msg.len() as u64);
            }
        }

        timed!(timing, "Write the execusion trace", {
            table.write_table_entries(&writer);
            blake_gadget.write(padded_messages, msg_lens.as_slice(), &writer, L::num_rows());
            let mut msg_to_check = 0;
            for i in 0..L::num_rows() {
                writer.write_row_instructions(&generator.air_data, i);

                let last_block_bit = writer.read(&blake_gadget.last_chunk_bit, i);
                let cycle_12_end_bit = writer.read(&blake_gadget.cycle_12_end_bit, i);
                if last_block_bit == F::ONE && cycle_12_end_bit == F::ONE {
                    let hash: [[GoldilocksField; HASH_ARRAY_SIZE]; 4] =
                        writer.read_array(&blake_gadget.h_output.get_subarray(0..8), i);
                    let calculated_hash_bytes = hash
                        .iter()
                        .flatten()
                        .map(|x| x.to_canonical_u64() as u8)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        calculated_hash_bytes.len(),
                        32,
                        "Hash should be 32 bytes long"
                    );
                    assert_eq!(
                        calculated_hash_bytes,
                        hex::decode(digests[msg_to_check]).unwrap(),
                        "Hashes do not match at row {}",
                        i
                    );

                    msg_to_check += 1;
                    msg_to_check %= msgs.len();
                }

                /* if i < 96 {
                    let mut v_input = Vec::new();
                    let mut v_output = Vec::new();
                    for j in 0..1 {
                        v_input.push(writer.read(&blake_gadget.v_input[j], i));
                        v_output.push(writer.read(&blake_gadget.v_output[j], i));
                    }
                    println!("i = {:?}, v_input = {:?}", i, v_input);
                    println!("i = {:?}, v_output = {:?}", i, v_output);

                    if i % 12 == 11 {
                        let h_output: [[GoldilocksField; 8]; 1] =
                            writer.read_array(&blake_gadget.h_output, i);
                        println!("i = {:?}, h_output = {:?}", i, h_output);

                        let hash_state: [[GoldilocksField; 8]; 1] = writer.read_array(
                            &blake_gadget
                                .hash_state
                                .get_subarray((i / 12) * 8..(i / 12) * 8 + 8),
                            0,
                        );
                        println!("i = {:?}, hash_state = {:?}", i, hash_state);
                    }
                }
                */
                /*
                if (i % 12) == 0 {
                    let msg: [[GoldilocksField; 8]; 1] = writer.read_array(
                        &blake_gadget
                            .msg_chunks
                            .get_subarray(i / 12 * 16..i / 12 * 16 + 1),
                        i,
                    );
                    println!("i = {:?}, msg = {:?}", i, msg);

                    let m: [[GoldilocksField; 8]; 1] = writer.read_array(&blake_gadget.m, i);
                    let unused_row = writer.read(&blake_gadget.unused_row, i);
                    println!("i = {:?}, m = {:?}", i, m);
                    println!("i = {:?}, unused_row = {:?}", i, unused_row);
                    let cycle_end_bit = writer.read(&blake_gadget.cycle_12_end_bit, i);
                    assert!(cycle_end_bit == F::ONE);
                }
                */
                /*
                if (i % 12) == 11 {
                    let hash: [[GoldilocksField; 8]; 1] = writer.read_array(
                        &blake_gadget
                            .hash_state
                            .get_subarray(i / 12 * 8..i / 12 * 8 + 1),
                        i,
                    );
                    println!("i = {:?}, hash = {:?}", i, hash);

                    let h: [[GoldilocksField; 8]; 1] = writer.read_array(&blake_gadget.h_output, i);
                    let unused_row = writer.read(&blake_gadget.unused_row, i);
                    println!("i = {:?}, h = {:?}", i, h);
                    println!("i = {:?}, unused_row = {:?}", i, unused_row);
                    let cycle_end_bit = writer.read(&blake_gadget.cycle_12_end_bit, i);
                    assert!(cycle_end_bit == F::ONE);
                }
                */

                /*
                let cycle_12_registers: [GoldilocksField; 12] =
                    writer.read_array(&blake_gadget.cycle_12_registers.get_subarray(0..12), i);
                println!("i = {:?}, cycle_12_registers = {:?}", i, cycle_12_registers);

                let padding_bit = writer.read(&blake_gadget.padding_bit, i);
                println!("i = {:?}, padding_bit = {:?}", i, padding_bit);
                */
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
