pub mod builder_gadget;
pub mod generator;

use core::borrow::Borrow;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use self::generator::BLAKE2BPublicData;
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
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub type U64Value<T> = <U64Register as Register>::Value<T>;

const NUM_MIX_ROUNDS: usize = 12;
const MSG_ARRAY_SIZE: usize = 16;
const HASH_ARRAY_SIZE: usize = 8;
const WORK_VECTOR_SIZE: usize = 16;
const CYCLE_12: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BLAKE2BGadget {
    pub padding_bit: BitRegister,
    pub cycle_12_start_bit: BitRegister,
    pub cycle_12_end_bit: BitRegister,

    pub m: ArrayRegister<U64Register>,
    pub t: U64Register,
    pub msg_last_chunk: BitRegister,
    pub msg_pad_row: BitRegister,
    pub max_last_row: BitRegister,
    pub h_input: ArrayRegister<U64Register>,
    pub h_output: ArrayRegister<U64Register>,
    pub unused_row: BitRegister,

    // Public values
    pub initial_hash: ArrayRegister<U64Register>,
    pub initial_hash_compress: ArrayRegister<U64Register>,
    pub inversion_const: U64Register,
    pub msg_chunks: ArrayRegister<U64Register>,
    pub t_public: ArrayRegister<U64Register>,
    pub msg_last_chunk_public: ArrayRegister<BitRegister>,
    pub msg_pad_chunk_public: ArrayRegister<BitRegister>,
    pub max_chunk_public: ArrayRegister<BitRegister>,
    pub hash_state: ArrayRegister<U64Register>,
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
    pub fn process_blake2b<const MAX_NUM_CHUNKS: usize>(
        &mut self,
        clk: &ElementRegister,
        bus: &mut Bus<CubicRegister, L::CubicParams>,
        bus_channel_idx: usize,
        operations: &mut ByteLookupOperations,
    ) -> BLAKE2BGadget
    where
        L::Instruction: UintInstructions,
    {
        let num_rows = MAX_NUM_CHUNKS * NUM_MIX_ROUNDS;
        assert!(num_rows <= (1 << 16));
        let num_chunks = MAX_NUM_CHUNKS;

        // Registers to be written to
        let m = self.alloc_array::<U64Register>(MSG_ARRAY_SIZE);
        let t = self.alloc::<U64Register>();
        let msg_last_chunk = self.alloc::<BitRegister>();
        // This is used to mark which rows are past a message's last chunk. E.g. if the max chunk
        // size is set to 10, and the input message is only 1 chunk, then the remaining 9 chunks's rows
        // will be marked with msg_pad_row.  For those rows, h_output should just be copied from the
        // previous row.  Note that the h_output values in the row WILL be part of the bus contraint.
        let msg_pad_row = self.alloc::<BitRegister>();
        let max_last_row = self.alloc::<BitRegister>();
        let h_input = self.alloc_array::<U64Register>(HASH_ARRAY_SIZE);
        let h_output = self.alloc_array::<U64Register>(HASH_ARRAY_SIZE);
        // This is used to mark which rows are not part of a message chunk.  E.g. if only a 10 chunk
        // message is submitted to curta, then the remaining rows' h_output columns will be set to 0.
        // Note that the h_output values in this row WILL be part of the bus contraint.
        let unused_row = self.alloc::<BitRegister>();
        // This is used to mark which rows pad the stark's trace to be a power of 2.  Since the chunk sizes
        // for blake have 12 rows, there will be 4 rows of padding.  Note that the h_output values
        // in this row WILL NOT be part of the bus contraint.
        let padding_bit = self.alloc::<BitRegister>();

        // Public values
        let initial_hash = self.alloc_array_public::<U64Register>(HASH_ARRAY_SIZE);
        let initial_hash_compress = self.alloc_array_public::<U64Register>(HASH_ARRAY_SIZE);
        let inversion_const = self.alloc_public::<U64Register>();
        let msg_chunks = self.alloc_array_public::<U64Register>(num_chunks * MSG_ARRAY_SIZE);
        let t_public = self.alloc_array_public::<U64Register>(num_chunks);
        let msg_last_chunk_public = self.alloc_array_public::<BitRegister>(num_chunks);
        let msg_pad_chunk_public = self.alloc_array_public::<BitRegister>(num_chunks);
        let max_chunk_public = self.alloc_array_public::<BitRegister>(num_chunks);
        let hash_state = self.alloc_array_public::<U64Register>(num_chunks * HASH_ARRAY_SIZE);

        let loop_iterations = self.loop_instr(CYCLE_12);
        let cycle_12_start_bit = loop_iterations.get_iteration_reg(0);
        let cycle_12_end_bit = loop_iterations.get_iteration_reg(CYCLE_12 - 1);

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
            &msg_last_chunk,
            &cycle_12_start_bit,
            &cycle_12_end_bit,
            &unused_row,
            &msg_pad_row,
            operations,
        );

        for ((h_in, init), h_out) in h_input.iter().zip(initial_hash.iter()).zip(h_output.iter()) {
            self.set_to_expression_transition(
                &h_in.next(),
                max_last_row.expr() * init.expr() + max_last_row.not_expr() * h_out.expr(),
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
            &msg_last_chunk,
            &msg_pad_row,
            &max_last_row,
            &h_output,
            &padding_bit,
            &msg_chunks,
            &t_public,
            &msg_last_chunk_public,
            &msg_pad_chunk_public,
            &max_chunk_public,
            &hash_state,
        );

        BLAKE2BGadget {
            padding_bit,
            cycle_12_start_bit,
            cycle_12_end_bit,

            m,
            t,
            msg_last_chunk,
            msg_pad_row,
            max_last_row,
            h_input,
            h_output,
            unused_row,

            initial_hash,
            initial_hash_compress,
            inversion_const,
            msg_chunks,
            t_public,
            msg_last_chunk_public,
            max_chunk_public,
            msg_pad_chunk_public,
            hash_state,
        }
    }

    // Note that we assume that t's max value is 2**64-1.
    // The blake2b specification actually does allow t to be up to 2**128-1
    // (see https://en.wikipedia.org/wiki/BLAKE_(hash_function)).
    // Also, we currently only support digest size of 32 bytes and no usage of a key.
    // This restricted support means that we can have a seperate constant for h0 in compress
    // instead of having to mutate it.
    #[allow(clippy::too_many_arguments)]
    fn blake2b_compress(
        &mut self,
        m: &ArrayRegister<U64Register>,
        h_input: &ArrayRegister<U64Register>,
        h_output: &ArrayRegister<U64Register>,
        iv_pub: &ArrayRegister<U64Register>,
        inversion_const_pub: &U64Register,
        t: &U64Register,
        msg_last_chunk: &BitRegister,
        cycle_12_start_bit: &BitRegister,
        cycle_12_end_bit: &BitRegister,
        unused_row: &BitRegister,
        msg_pad_row: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: UintInstructions,
    {
        let v_compress_init = self.blake2b_compress_initialize(
            iv_pub,
            inversion_const_pub,
            h_input,
            t,
            msg_last_chunk,
            operations,
        );

        // This is set to the previous row's v_mix_output.
        // Note that this array will not be read for the first row of the compress cycle.
        // v_compress_init will be read instead
        let v_mix_input = self.alloc_array::<U64Register>(WORK_VECTOR_SIZE);

        let mut v_work_vec_0 =
            self.select(cycle_12_start_bit, &v_compress_init[0], &v_mix_input.get(0));
        let mut v_work_vec_1 =
            self.select(cycle_12_start_bit, &v_compress_init[1], &v_mix_input.get(1));
        let mut v_work_vec_2 =
            self.select(cycle_12_start_bit, &v_compress_init[2], &v_mix_input.get(2));
        let mut v_work_vec_3 =
            self.select(cycle_12_start_bit, &v_compress_init[3], &v_mix_input.get(3));
        let mut v_work_vec_4 =
            self.select(cycle_12_start_bit, &v_compress_init[4], &v_mix_input.get(4));
        let mut v_work_vec_5 =
            self.select(cycle_12_start_bit, &v_compress_init[5], &v_mix_input.get(5));
        let mut v_work_vec_6 =
            self.select(cycle_12_start_bit, &v_compress_init[6], &v_mix_input.get(6));
        let mut v_work_vec_7 =
            self.select(cycle_12_start_bit, &v_compress_init[7], &v_mix_input.get(7));
        let mut v_work_vec_8 =
            self.select(cycle_12_start_bit, &v_compress_init[8], &v_mix_input.get(8));
        let mut v_work_vec_9 =
            self.select(cycle_12_start_bit, &v_compress_init[9], &v_mix_input.get(9));
        let mut v_work_vec_10 = self.select(
            cycle_12_start_bit,
            &v_compress_init[10],
            &v_mix_input.get(10),
        );
        let mut v_work_vec_11 = self.select(
            cycle_12_start_bit,
            &v_compress_init[11],
            &v_mix_input.get(11),
        );
        let mut v_work_vec_12 = self.select(
            cycle_12_start_bit,
            &v_compress_init[12],
            &v_mix_input.get(12),
        );
        let mut v_work_vec_13 = self.select(
            cycle_12_start_bit,
            &v_compress_init[13],
            &v_mix_input.get(13),
        );
        let mut v_work_vec_14 = self.select(
            cycle_12_start_bit,
            &v_compress_init[14],
            &v_mix_input.get(14),
        );
        let mut v_work_vec_15 = self.select(
            cycle_12_start_bit,
            &v_compress_init[15],
            &v_mix_input.get(15),
        );

        let v_mix_output = self.alloc_array::<U64Register>(WORK_VECTOR_SIZE);
        self.blake2b_mix_rounds(
            &mut v_work_vec_0,
            &mut v_work_vec_1,
            &mut v_work_vec_2,
            &mut v_work_vec_3,
            &mut v_work_vec_4,
            &mut v_work_vec_5,
            &mut v_work_vec_6,
            &mut v_work_vec_7,
            &mut v_work_vec_8,
            &mut v_work_vec_9,
            &mut v_work_vec_10,
            &mut v_work_vec_11,
            &mut v_work_vec_12,
            &mut v_work_vec_13,
            &mut v_work_vec_14,
            &mut v_work_vec_15,
            m,
            &v_mix_output,
            operations,
        );

        self.set_to_expression_transition(&v_mix_input.get(0).next(), v_mix_output.get(0).expr());
        self.set_to_expression_transition(&v_mix_input.get(1).next(), v_mix_output.get(1).expr());
        self.set_to_expression_transition(&v_mix_input.get(2).next(), v_mix_output.get(2).expr());
        self.set_to_expression_transition(&v_mix_input.get(3).next(), v_mix_output.get(3).expr());
        self.set_to_expression_transition(&v_mix_input.get(4).next(), v_mix_output.get(4).expr());
        self.set_to_expression_transition(&v_mix_input.get(5).next(), v_mix_output.get(5).expr());
        self.set_to_expression_transition(&v_mix_input.get(6).next(), v_mix_output.get(6).expr());
        self.set_to_expression_transition(&v_mix_input.get(7).next(), v_mix_output.get(7).expr());
        self.set_to_expression_transition(&v_mix_input.get(8).next(), v_mix_output.get(8).expr());
        self.set_to_expression_transition(&v_mix_input.get(9).next(), v_mix_output.get(9).expr());
        self.set_to_expression_transition(&v_mix_input.get(10).next(), v_mix_output.get(10).expr());
        self.set_to_expression_transition(&v_mix_input.get(11).next(), v_mix_output.get(11).expr());
        self.set_to_expression_transition(&v_mix_input.get(12).next(), v_mix_output.get(12).expr());
        self.set_to_expression_transition(&v_mix_input.get(13).next(), v_mix_output.get(13).expr());
        self.set_to_expression_transition(&v_mix_input.get(14).next(), v_mix_output.get(14).expr());
        self.set_to_expression_transition(&v_mix_input.get(15).next(), v_mix_output.get(15).expr());

        let mut h_0_tmp = self.bitwise_xor(&h_input.get(0), &v_mix_output.get(0), operations);
        let mut h_1_tmp = self.bitwise_xor(&h_input.get(1), &v_mix_output.get(1), operations);
        let mut h_2_tmp = self.bitwise_xor(&h_input.get(2), &v_mix_output.get(2), operations);
        let mut h_3_tmp = self.bitwise_xor(&h_input.get(3), &v_mix_output.get(3), operations);
        let mut h_4_tmp = self.bitwise_xor(&h_input.get(4), &v_mix_output.get(4), operations);
        let mut h_5_tmp = self.bitwise_xor(&h_input.get(5), &v_mix_output.get(5), operations);
        let mut h_6_tmp = self.bitwise_xor(&h_input.get(6), &v_mix_output.get(6), operations);
        let mut h_7_tmp = self.bitwise_xor(&h_input.get(7), &v_mix_output.get(7), operations);

        h_0_tmp = self.bitwise_xor(&h_0_tmp, &v_mix_output.get(8), operations);
        h_1_tmp = self.bitwise_xor(&h_1_tmp, &v_mix_output.get(9), operations);
        h_2_tmp = self.bitwise_xor(&h_2_tmp, &v_mix_output.get(10), operations);
        h_3_tmp = self.bitwise_xor(&h_3_tmp, &v_mix_output.get(11), operations);
        h_4_tmp = self.bitwise_xor(&h_4_tmp, &v_mix_output.get(12), operations);
        h_5_tmp = self.bitwise_xor(&h_5_tmp, &v_mix_output.get(13), operations);
        h_6_tmp = self.bitwise_xor(&h_6_tmp, &v_mix_output.get(14), operations);
        h_7_tmp = self.bitwise_xor(&h_7_tmp, &v_mix_output.get(15), operations);

        let u64_register_zero = ArithmeticExpression::from_constant_vec(vec![L::Field::ZERO; 8]);

        // This is set to true if not in last bit of cycle_12 || within the msg pad rows
        let h_no_op_row = self.alloc::<BitRegister>();
        self.set_to_expression(
            &h_no_op_row,
            (cycle_12_end_bit.not_expr() + msg_pad_row.expr())
                - (cycle_12_end_bit.not_expr() * msg_pad_row.expr()),
        );

        // set h_output to zero if it's an unused row
        self.set_to_expression(
            &h_output.get(0),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_0_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(0).expr())),
        );
        self.set_to_expression(
            &h_output.get(1),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_1_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(1).expr())),
        );
        self.set_to_expression(
            &h_output.get(2),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_2_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(2).expr())),
        );
        self.set_to_expression(
            &h_output.get(3),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_3_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(3).expr())),
        );
        self.set_to_expression(
            &h_output.get(4),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_4_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(4).expr())),
        );
        self.set_to_expression(
            &h_output.get(5),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_5_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(5).expr())),
        );
        self.set_to_expression(
            &h_output.get(6),
            unused_row.expr() * u64_register_zero.clone()
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_6_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(6).expr())),
        );
        self.set_to_expression(
            &h_output.get(7),
            unused_row.expr() * u64_register_zero
                + (unused_row.not_expr()
                    * (h_no_op_row.not_expr() * h_7_tmp.expr()
                        + h_no_op_row.expr() * h_input.get(7).expr())),
        );
    }

    // As noted before, we assume that t's max value is 2**64-1.
    // That assumption and restricted support means that we don't have to mutate V13.
    fn blake2b_compress_initialize(
        &mut self,
        iv_pub: &ArrayRegister<U64Register>,
        inversion_const_pub: &U64Register,
        h_input: &ArrayRegister<U64Register>,
        t: &U64Register,
        last_chunk_bit: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) -> [U64Register; WORK_VECTOR_SIZE]
    where
        L::Instruction: UintInstructions,
    {
        // Need to create non public registers for IV and inversion_const.
        // Operations that use both public and private registers causes issues.
        let iv = self.alloc_array::<U64Register>(HASH_ARRAY_SIZE);
        for i in 0..HASH_ARRAY_SIZE {
            self.set_to_expression(&iv.get(i), iv_pub.get(i).expr());
        }

        let inversion_const = self.alloc::<U64Register>();
        self.set_to_expression(&inversion_const, inversion_const_pub.expr());

        // Allocate v_compress_input.
        // This is read only on the first row of cycle 12
        let v_compress_input_0 = &h_input.get(0);
        let v_compress_input_1 = &h_input.get(1);
        let v_compress_input_2 = &h_input.get(2);
        let v_compress_input_3 = &h_input.get(3);
        let v_compress_input_4 = &h_input.get(4);
        let v_compress_input_5 = &h_input.get(5);
        let v_compress_input_6 = &h_input.get(6);
        let v_compress_input_7 = &h_input.get(7);
        let v_compress_input_8 = &iv.get(0);
        let v_compress_input_9 = &iv.get(1);
        let v_compress_input_10 = &iv.get(2);
        let v_compress_input_11 = &iv.get(3);
        let mut v_compress_input_12 = &iv.get(4);
        let v_compress_input_13 = &iv.get(5);
        let mut v_compress_input_14 = &iv.get(6);
        let v_compress_input_15 = &iv.get(7);

        // If this is the first bit of the 12 round mix cycle, initialize the work vector, else set it to v_out
        let v_12_t = self.bitwise_xor(v_compress_input_12, t, operations);
        v_compress_input_12 = &v_12_t;
        // We assume that the t is not more than u64, so we don't modify v_compress_input_13

        // Invert v[14] bits if this is the last block and we are at the start of the mix 12 cycle.
        let v_14_inverted = &self.bitwise_xor(v_compress_input_14, &inversion_const, operations);
        let v_compress_input_v14_last_block =
            self.select::<U64Register>(last_chunk_bit, v_14_inverted, v_compress_input_14);
        v_compress_input_14 = &v_compress_input_v14_last_block;

        [
            *v_compress_input_0,
            *v_compress_input_1,
            *v_compress_input_2,
            *v_compress_input_3,
            *v_compress_input_4,
            *v_compress_input_5,
            *v_compress_input_6,
            *v_compress_input_7,
            *v_compress_input_8,
            *v_compress_input_9,
            *v_compress_input_10,
            *v_compress_input_11,
            *v_compress_input_12,
            *v_compress_input_13,
            *v_compress_input_14,
            *v_compress_input_15,
        ]
    }

    #[allow(clippy::too_many_arguments)]
    fn blake2b_mix_rounds<'a>(
        &'a mut self,
        v_0: &'a mut U64Register,
        v_1: &'a mut U64Register,
        v_2: &'a mut U64Register,
        v_3: &'a mut U64Register,
        v_4: &'a mut U64Register,
        v_5: &'a mut U64Register,
        v_6: &'a mut U64Register,
        v_7: &'a mut U64Register,
        v_8: &'a mut U64Register,
        v_9: &'a mut U64Register,
        v_10: &'a mut U64Register,
        v_11: &'a mut U64Register,
        v_12: &'a mut U64Register,
        v_13: &'a mut U64Register,
        v_14: &'a mut U64Register,
        v_15: &'a mut U64Register,
        m: &ArrayRegister<U64Register>,
        v_output: &ArrayRegister<U64Register>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: UintInstructions,
    {
        self.blake2b_mix(v_0, v_4, v_8, v_12, &m.get(0), &m.get(1), operations);

        self.blake2b_mix(v_1, v_5, v_9, v_13, &m.get(2), &m.get(3), operations);

        self.blake2b_mix(v_2, v_6, v_10, v_14, &m.get(4), &m.get(5), operations);

        self.blake2b_mix(v_3, v_7, v_11, v_15, &m.get(6), &m.get(7), operations);

        self.blake2b_mix(v_0, v_5, v_10, v_15, &m.get(8), &m.get(9), operations);

        self.blake2b_mix(v_1, v_6, v_11, v_12, &m.get(10), &m.get(11), operations);

        self.blake2b_mix(v_2, v_7, v_8, v_13, &m.get(12), &m.get(13), operations);

        self.blake2b_mix(v_3, v_4, v_9, v_14, &m.get(14), &m.get(15), operations);

        self.set_to_expression(&v_output.get(0), v_0.expr());
        self.set_to_expression(&v_output.get(1), v_1.expr());
        self.set_to_expression(&v_output.get(2), v_2.expr());
        self.set_to_expression(&v_output.get(3), v_3.expr());
        self.set_to_expression(&v_output.get(4), v_4.expr());
        self.set_to_expression(&v_output.get(5), v_5.expr());
        self.set_to_expression(&v_output.get(6), v_6.expr());
        self.set_to_expression(&v_output.get(7), v_7.expr());
        self.set_to_expression(&v_output.get(8), v_8.expr());
        self.set_to_expression(&v_output.get(9), v_9.expr());
        self.set_to_expression(&v_output.get(10), v_10.expr());
        self.set_to_expression(&v_output.get(11), v_11.expr());
        self.set_to_expression(&v_output.get(12), v_12.expr());
        self.set_to_expression(&v_output.get(13), v_13.expr());
        self.set_to_expression(&v_output.get(14), v_14.expr());
        self.set_to_expression(&v_output.get(15), v_15.expr());
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
        L::Instruction: UintInstructions,
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

    #[allow(clippy::too_many_arguments)]
    pub fn add_bus_constraints(
        &mut self,
        clk: &ElementRegister,
        bus: &mut Bus<CubicRegister, L::CubicParams>,
        bus_channel_idx: usize,
        num_chunks: usize,
        cycle_12_start_bit: &BitRegister,
        cycle_12_end_bit: &BitRegister,

        m: &ArrayRegister<U64Register>,
        t: &U64Register,
        msg_last_chunk: &BitRegister,
        msg_pad_row: &BitRegister,
        max_last_row: &BitRegister,
        h_output: &ArrayRegister<U64Register>,
        padding_bit: &BitRegister,

        msg_chunks: &ArrayRegister<U64Register>,
        t_public: &ArrayRegister<U64Register>,
        msg_last_chunk_public: &ArrayRegister<BitRegister>,
        msg_pad_chunk_public: &ArrayRegister<BitRegister>,
        max_chunk_public: &ArrayRegister<BitRegister>,
        hash_state: &ArrayRegister<U64Register>,
    ) where
        L::Instruction: UintInstructions,
    {
        // Get message chunk challenges
        let message_chunk_challenges = self
            .alloc_array_challenge::<CubicRegister>(U64Register::size_of() * MSG_ARRAY_SIZE + 1);

        // Get hash state challenges
        let state_challenges = self
            .alloc_array_challenge::<CubicRegister>(U64Register::size_of() * HASH_ARRAY_SIZE + 1);

        // Get the msg last chunk callenges
        let msg_last_chunk_challenges = self.alloc_array_challenge::<CubicRegister>(2);

        // Get the msg pad chunk challenges
        let msg_pad_chunk_challenges = self.alloc_array_challenge::<CubicRegister>(2);

        // Get the max chunk challenges
        let max_chunk_challenges = self.alloc_array_challenge::<CubicRegister>(2);

        // Get the t challenges
        let t_challenges = self.alloc_array_challenge::<CubicRegister>(U64Register::size_of() + 1);

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

                // For the last row of every cycle, send the public hash state to the bus and the
                // max_chunk bit
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

                    let max_chunk_digest = self.accumulate_public_expressions(
                        &max_chunk_challenges,
                        &[row_expr.clone(), max_chunk_public.get(i).expr()],
                    );
                    bus.output_global_value(&max_chunk_digest);
                }

                // For every row of the cycle, send the msg last chunk bit to the bus
                let msg_last_chunk_digest = self.accumulate_public_expressions(
                    &msg_last_chunk_challenges,
                    &[row_expr.clone(), msg_last_chunk_public.get(i).expr()],
                );

                bus.output_global_value(&msg_last_chunk_digest);

                // For every row of the cycle, send the msg pad chunk bit to the bus
                let msg_pad_chunk_digest = self.accumulate_public_expressions(
                    &msg_pad_chunk_challenges,
                    &[row_expr.clone(), msg_pad_chunk_public.get(i).expr()],
                );
                bus.output_global_value(&msg_pad_chunk_digest);

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

        let not_padding = self.alloc::<BitRegister>();
        self.set_to_expression(&not_padding, padding_bit.not_expr());

        let cycle_12_start_no_padding = self.alloc::<BitRegister>();
        self.set_to_expression(
            &cycle_12_start_no_padding,
            cycle_12_start_bit.expr() * not_padding.expr(),
        );

        let cycle_12_end_no_padding = self.alloc::<BitRegister>();
        self.set_to_expression(
            &cycle_12_end_no_padding,
            cycle_12_end_bit.expr() * not_padding.expr(),
        );

        let t_digest = self.accumulate_expressions(&t_challenges, &[clk.expr(), t.expr()]);
        self.input_to_bus_filtered(bus_channel_idx, t_digest, cycle_12_start_no_padding);
        self.assert_expression_zero(
            t.expr() * cycle_12_start_bit.not_expr() * padding_bit.not_expr(),
        );

        let clk_hash_next = self.accumulate_expressions(
            &state_challenges,
            &[clk.expr(), h_output.get_subarray(0..HASH_ARRAY_SIZE).expr()],
        );
        self.input_to_bus_filtered(bus_channel_idx, clk_hash_next, cycle_12_end_no_padding);

        let clk_msg_last_chunk_digest = self.accumulate_expressions(
            &msg_last_chunk_challenges,
            &[clk.expr(), msg_last_chunk.expr()],
        );
        self.input_to_bus_filtered(bus_channel_idx, clk_msg_last_chunk_digest, not_padding);

        let clk_msg_pad_chunk_digest = self
            .accumulate_expressions(&msg_pad_chunk_challenges, &[clk.expr(), msg_pad_row.expr()]);
        self.input_to_bus_filtered(bus_channel_idx, clk_msg_pad_chunk_digest, not_padding);

        let clk_max_row_digest =
            self.accumulate_expressions(&max_chunk_challenges, &[clk.expr(), max_last_row.expr()]);
        self.input_to_bus_filtered(bus_channel_idx, clk_max_row_digest, cycle_12_end_no_padding);

        let clk_msg_digest = self.accumulate_expressions(
            &message_chunk_challenges,
            &[clk.expr(), m.get_subarray(0..MSG_ARRAY_SIZE).expr()],
        );
        self.input_to_bus_filtered(bus_channel_idx, clk_msg_digest, not_padding);
    }
}

impl BLAKE2BGadget {
    pub fn write<F: Field, I: IntoIterator>(
        &self,
        padded_messages: I,
        message_lens: &[u64],
        max_chunk_sizes: &[u64],
        writer: &TraceWriter<F>,
        num_rows: usize,
    ) -> BLAKE2BPublicData<F>
    where
        I::Item: Borrow<[u8]>,
    {
        let max_total_num_chunks = num_rows / NUM_MIX_ROUNDS;

        // First check that sum max_chunk_sizes <= max_total_num_chunks
        assert!(max_chunk_sizes.iter().sum::<u64>() <= max_total_num_chunks as u64);

        // Public values
        let mut hash_values_public = Vec::new();
        let mut msg_chunks_public = Vec::<[F; 8]>::new();
        let mut msg_last_chunk_public = Vec::new();
        let mut msg_pad_chunk_public = Vec::new();
        let mut max_chunk_public = Vec::new();
        let mut t_values_public = Vec::new();

        let mut num_written_chunks = 0usize;
        let mut row_num = 0;
        for ((padded_msg, message_len), max_chunk_size) in padded_messages
            .into_iter()
            .zip_eq(message_lens.iter())
            .zip_eq(max_chunk_sizes.iter())
        {
            let padded_msg = padded_msg.borrow();

            let mut msg_num_chunks = *message_len / 128;
            if (*message_len % 128 != 0) || (*message_len == 0) {
                msg_num_chunks += 1;
            }

            assert!(padded_msg.len() % 128 == 0, "Message not padded correctly");
            assert!(msg_num_chunks <= *max_chunk_size);

            let mut state = INITIAL_HASH;
            let mut bytes_compressed = 0;

            for chunk_num in 0..*max_chunk_size {
                let msg_last_chunk = chunk_num == msg_num_chunks - 1;
                let msg_last_chunk_value = if msg_last_chunk { F::ONE } else { F::ZERO };
                let max_chunk = chunk_num == *max_chunk_size - 1;
                let max_chunk_value = if max_chunk { F::ONE } else { F::ZERO };
                let past_last_chunk = chunk_num >= msg_num_chunks;
                let pad_chunk_value = if past_last_chunk { F::ONE } else { F::ZERO };

                let mut chunk = [0u8; 128];
                if !past_last_chunk {
                    chunk.copy_from_slice(
                        &padded_msg[chunk_num as usize * 128..(chunk_num as usize + 1) * 128],
                    );
                }

                if msg_last_chunk || past_last_chunk {
                    bytes_compressed = *message_len;
                } else {
                    bytes_compressed += 128;
                }

                msg_last_chunk_public.push(msg_last_chunk_value);
                msg_pad_chunk_public.push(pad_chunk_value);
                max_chunk_public.push(max_chunk_value);

                writer.write(
                    &self.t,
                    &u64_to_le_field_bytes::<F>(bytes_compressed),
                    row_num,
                );
                t_values_public.push(u64_to_le_field_bytes::<F>(bytes_compressed));

                if !past_last_chunk {
                    state = BLAKE2BGadget::compress(
                        &chunk,
                        &mut state,
                        bytes_compressed,
                        msg_last_chunk,
                    );
                }

                hash_values_public.extend_from_slice(&state.map(u64_to_le_field_bytes::<F>));

                // Get the message chunk array
                let chunk_array: [[F; 8]; MSG_ARRAY_SIZE] = chunk
                    .chunks_exact(8)
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|y| F::from_canonical_u8(*y))
                            .collect::<Vec<_>>()
                            .as_slice()
                            .try_into()
                            .expect("should be slice of 8 elements")
                    })
                    .collect::<Vec<_>>()
                    .as_slice()
                    .try_into()
                    .expect("should be slice of 16 elements");

                msg_chunks_public.extend_from_slice(&chunk_array);

                // Write the last_chunk_value, msg_pad_row_value, max_chunk_value, and m columns
                for i in 0..NUM_MIX_ROUNDS {
                    let permuted_chunk_array = self.permute_msgs(&chunk_array, i);
                    writer.write_array(&self.m, &permuted_chunk_array, row_num);

                    writer.write(&self.msg_last_chunk, &msg_last_chunk_value, row_num);

                    if past_last_chunk {
                        writer.write(&self.msg_pad_row, &F::ONE, row_num);
                    }

                    if i == NUM_MIX_ROUNDS - 1 && max_chunk {
                        writer.write(&self.max_last_row, &F::ONE, row_num);
                    }

                    row_num += 1;
                }

                num_written_chunks += 1;
            }
        }

        assert!(hash_values_public.len() == num_written_chunks * 8);
        assert!(
            msg_last_chunk_public
                .iter()
                .filter(|x| **x == F::ONE)
                .count()
                == message_lens.len()
        );
        assert!(max_chunk_public.iter().filter(|x| **x == F::ONE).count() == max_chunk_sizes.len());

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

        // pad msg_chunks_public to max_total_num_chunks * MSG_ARRAY_SIZE
        msg_chunks_public.extend(vec![
            [F::ZERO; 8];
            (max_total_num_chunks - num_written_chunks)
                * MSG_ARRAY_SIZE
        ]);
        writer.write_array(&self.msg_chunks, &msg_chunks_public, 0);

        // pad t_values_public to max_total_num_chunks
        t_values_public.extend(vec![
            [F::ZERO; 8];
            max_total_num_chunks - num_written_chunks
        ]);
        writer.write_array(&self.t_public, &t_values_public, 0);

        // pad msg_last_chunk_public to max_total_num_chunks
        msg_last_chunk_public.extend(vec![F::ZERO; max_total_num_chunks - num_written_chunks]);
        writer.write_array(&self.msg_last_chunk_public, &msg_last_chunk_public, 0);

        // pad msg_pad_chunk_public to max_total_num_chunks
        msg_pad_chunk_public.extend(vec![F::ZERO; max_total_num_chunks - num_written_chunks]);
        writer.write_array(&self.msg_pad_chunk_public, &msg_pad_chunk_public, 0);

        // pad max_chunk_public to max_total_num_chunks
        max_chunk_public.extend(vec![F::ZERO; max_total_num_chunks - num_written_chunks]);
        writer.write_array(&self.max_chunk_public, &max_chunk_public, 0);

        // pad hash_values_public to max_total_num_chunks * HASH_ARRAY_SIZE
        hash_values_public.extend(vec![
            [F::ZERO; 8];
            (max_total_num_chunks - num_written_chunks)
                * HASH_ARRAY_SIZE
        ]);
        writer.write_array(&self.hash_state, &hash_values_public, 0);

        let num_padding_bits = num_rows % NUM_MIX_ROUNDS;

        // Need to pad the rest of the rows
        for i in num_written_chunks * NUM_MIX_ROUNDS..num_rows {
            writer.write(&self.unused_row, &F::ONE, i);

            if i >= num_rows - num_padding_bits {
                writer.write(&self.padding_bit, &F::ONE, i);
            } else {
                writer.write(&self.padding_bit, &F::ZERO, i);
            }
        }

        for i in num_rows..1 << 16 {
            writer.write(&self.unused_row, &F::ONE, i);
            writer.write(&self.padding_bit, &F::ONE, i);
        }

        BLAKE2BPublicData {
            msg_chunks: msg_chunks_public,
            t: t_values_public,
            msg_last_chunk: msg_last_chunk_public,
            msg_pad_chunk: msg_pad_chunk_public,
            max_chunk: max_chunk_public,
            hash_state: hash_values_public,
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
    ) -> [u64; HASH_ARRAY_SIZE] {
        // Set up the work vector V
        let mut v: [u64; WORK_VECTOR_SIZE] = [0; WORK_VECTOR_SIZE];

        v[..8].copy_from_slice(&state[..HASH_ARRAY_SIZE]);
        v[8..16].copy_from_slice(&INITIAL_HASH_COMPRESS);

        v[12] ^= bytes_compressed;
        if last_chunk {
            v[14] ^= INVERSION_CONST;
        }

        let msg_u64_chunks = msg_chunk
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();

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

        for i in 0..HASH_ARRAY_SIZE {
            state[i] ^= v[i];
        }

        for i in 0..HASH_ARRAY_SIZE {
            state[i] ^= v[i + 8];
        }

        *state
    }

    fn mix(
        v: &mut [u64; WORK_VECTOR_SIZE],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x: u64,
        y: u64,
    ) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }

    pub fn pad(msg: &[u8], max_chunk_size: u64) -> Vec<u8> {
        let mut msg_chunk_size = msg.len() as u64 / 128;

        if (msg.len() % 128 != 0) || msg.is_empty() {
            msg_chunk_size += 1;
        }

        assert!(msg_chunk_size <= max_chunk_size, "Message too big");

        let padlen = max_chunk_size * 128 - msg.len() as u64;
        if padlen > 0 {
            let mut padded_msg = Vec::new();
            padded_msg.extend_from_slice(msg);
            padded_msg.extend_from_slice(&vec![0u8; padlen as usize]);
            padded_msg
        } else {
            msg.to_vec()
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
    use crate::chip::hash::blake::blake2b::generator::BLAKE2BAirParameters;

    #[test]
    fn test_blake2b_stark() {
        type F = GoldilocksField;
        type E = GoldilocksCubicParameters;
        type L = BLAKE2BAirParameters<F, E>;
        type SC = PoseidonGoldilocksStarkConfig;

        const MAX_NUM_CHUNKS: usize = 1365;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Blake2b test", log::Level::Info);

        let mut builder = AirBuilder::<L>::new();
        let clk = builder.clock();

        let mut operations = builder.byte_operations();

        let mut bus = builder.new_bus();
        let channel_idx = bus.new_channel(&mut builder);

        let blake_gadget =
            builder.process_blake2b::<MAX_NUM_CHUNKS>(&clk, &mut bus, channel_idx, &mut operations);

        let mut byte_table = builder.new_byte_lookup_table();
        let byte_data = builder.register_byte_lookup(&mut byte_table, operations);
        builder.constraint_byte_lookup_table(&byte_table);
        builder.constrain_bus(bus);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
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
        let msg_max_chunk_sizes = [4u64, 4, 35, 35];

        let digests = [
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8",
            "dad415aa819ebb585ce8ee1c1fa883804f405f6d8a6a0992628fb3bdaab5b42e",
            "022bfe46002fe82ab0c451574898fafaeb36283825aab39ddf825dc48a1c0970",
            "ad58001fb7de22fada15e574b17b2d2485183320cdc14f78625574328fedba84",
        ];

        let mut padded_messages = Vec::new();
        let mut msg_lens = Vec::new();
        let mut max_chunk_sizes = Vec::new();

        for _i in 0..17 {
            for (msg, msg_max_chunk_size) in msgs.iter().zip(msg_max_chunk_sizes.iter()) {
                padded_messages.push(
                    BLAKE2BGadget::pad(msg, *msg_max_chunk_size)
                        .into_iter()
                        .collect::<Vec<_>>(),
                );
                msg_lens.push(msg.len() as u64);
                max_chunk_sizes.push(*msg_max_chunk_size);
            }
        }

        timed!(timing, log::Level::Info, "Write the execusion trace", {
            byte_table.write_table_entries(&writer);
            blake_gadget.write(
                padded_messages,
                msg_lens.as_slice(),
                max_chunk_sizes.as_slice(),
                &writer,
                num_rows / 4,
            );
            let mut msg_to_check = 0;
            for i in 0..num_rows {
                writer.write_row_instructions(&generator.air_data, i);

                let msg_last_chunk = writer.read(&blake_gadget.msg_last_chunk, i);
                let cycle_12_end_bit = writer.read(&blake_gadget.cycle_12_end_bit, i);
                if msg_last_chunk == F::ONE && cycle_12_end_bit == F::ONE {
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
            }
            let multiplicities = byte_data.get_multiplicities(&writer);
            writer.write_lookup_multiplicities(byte_table.multiplicities(), &[multiplicities]);
        });

        let public_inputs = writer.0.public.read().unwrap().clone();
        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        timed!(
            timing,
            log::Level::Info,
            "Stark proof and verify",
            test_starky(&stark, &config, &generator, &public_inputs)
        );

        // Generate recursive proof
        timed!(
            timing,
            log::Level::Info,
            "Recursive proof generation and verification",
            test_recursive_starky(stark, config, generator, &public_inputs)
        );

        timing.print();
    }
}
