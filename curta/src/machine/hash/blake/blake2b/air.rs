use log::debug;
use plonky2::util::log2_ceil;

use super::data::{BLAKE2BConstNums, BLAKE2BConsts, BLAKE2BData};
use super::{BLAKE2BAir, IV};
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::machine::hash::blake::blake2b::data::{
    BLAKE2BMemory, BLAKE2BPublicData, BLAKE2BTraceData, MemoryArray,
};
use crate::machine::hash::blake::blake2b::{
    COMPRESS_IV, SIGMA_PERMUTATIONS, V_INDICES, V_LAST_WRITE_AGES,
};
use crate::math::prelude::*;

const DUMMY_INDEX: u64 = i32::MAX as u64;
const DUMMY_TS: u64 = (i32::MAX - 1) as u64;
const FIRST_COMPRESS_H_READ_TS: u64 = i32::MAX as u64;

impl<L: AirParameters> BLAKE2BAir<L>
where
    L::Instruction: UintInstructions,
{
    fn cycles_end_bits(
        builder: &mut BytesBuilder<L>,
    ) -> (BitRegister, BitRegister, BitRegister, BitRegister) {
        let cycle_4 = builder.cycle(2);
        let cycle_8 = builder.cycle(3);
        let loop_3 = builder.api.loop_instr(3);
        let cycle_96_end_bit = {
            let cycle_32 = builder.cycle(5);
            builder.mul(loop_3.get_iteration_reg(2), cycle_32.end_bit)
        };

        (
            loop_3.get_iteration_reg(2),
            cycle_4.end_bit,
            cycle_8.end_bit,
            cycle_96_end_bit,
        )
    }

    pub fn blake2b(
        builder: &mut BytesBuilder<L>,
        padded_chunks: &[ArrayRegister<U64Register>],
        t_values: &ArrayRegister<U64Register>,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: &ArrayRegister<ElementRegister>,
        num_messages: &ElementRegister,
    ) -> Vec<ArrayRegister<U64Register>> {
        let data = Self::blake2b_data(
            builder,
            padded_chunks,
            t_values,
            end_bits,
            digest_bits,
            digest_indices,
            num_messages,
        );

        let (v_indices, v_values) = Self::blake2b_compress_initialize(builder, &data);
        Self::blake2b_compress(builder, &v_indices, &v_values, &data)
    }

    pub fn blake2b_const_nums(builder: &mut BytesBuilder<L>) -> BLAKE2BConstNums {
        BLAKE2BConstNums {
            const_0: builder.constant(&L::Field::from_canonical_u8(0)),
            const_0_u64: builder.constant(&u64_to_le_field_bytes(0u64)),
            const_1: builder.constant(&L::Field::from_canonical_u8(1)),
            const_2: builder.constant(&L::Field::from_canonical_u8(2)),
            const_3: builder.constant(&L::Field::from_canonical_u8(3)),
            const_4: builder.constant(&L::Field::from_canonical_u8(4)),
            const_8: builder.constant(&L::Field::from_canonical_u8(8)),
            const_10: builder.constant(&L::Field::from_canonical_u8(10)),
            const_16: builder.constant(&L::Field::from_canonical_u8(16)),
            const_24: builder.constant(&L::Field::from_canonical_u8(24)),
            const_91: builder.constant(&L::Field::from_canonical_u8(91)),
            const_95: builder.constant(&L::Field::from_canonical_u8(95)),
            const_96: builder.constant(&L::Field::from_canonical_u8(96)),
            const_97: builder.constant(&L::Field::from_canonical_u8(97)),
            const_184: builder.constant(&L::Field::from_canonical_u8(184)),
            const_ffffffffffffffff: builder.constant::<U64Register>(&u64_to_le_field_bytes::<
                L::Field,
            >(
                0xFFFFFFFFFFFFFFFF
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn blake2b_const(
        builder: &mut BytesBuilder<L>,
        num_rows: &ElementRegister,
        num_messages: &ElementRegister,
        num_compresses: &ElementRegister,
        num_mix_iterations: usize,
        num_mix_iterations_last_round: usize,
        const_nums: &BLAKE2BConstNums,
    ) -> BLAKE2BConsts<L> {
        assert!(DUMMY_INDEX < L::Field::order());
        let dummy_index: ElementRegister =
            builder.constant(&L::Field::from_canonical_u64(DUMMY_INDEX));

        assert!(DUMMY_TS < L::Field::order());
        let dummy_ts: ElementRegister = builder.constant(&L::Field::from_canonical_u64(DUMMY_TS));

        assert!(FIRST_COMPRESS_H_READ_TS < L::Field::order());
        let first_compress_h_read_ts: ElementRegister =
            builder.constant(&L::Field::from_canonical_u64(FIRST_COMPRESS_H_READ_TS));

        let iv_values = builder.constant_array::<U64Register>(&IV.map(u64_to_le_field_bytes));
        let iv: Slice<crate::chip::uint::register::ByteArrayRegister<8>> = builder.uninit_slice();
        let num_messages_plus_1 = builder.alloc_public::<ElementRegister>();
        builder.set_to_expression(
            &num_messages_plus_1,
            num_messages.expr() + const_nums.const_1.expr(),
        );
        for (i, value) in iv_values.iter().enumerate() {
            builder.store(&iv.get(i), value, &Time::zero(), Some(num_messages_plus_1));
        }
        // The dummy iv value is read twice at the rows other than the first 4 rows of the each messages's
        // first compress round.
        let num_dummy_iv_reads = builder.alloc_public::<ElementRegister>();
        builder.set_to_expression(
            &num_dummy_iv_reads,
            (num_rows.expr() - (num_messages_plus_1.expr() * const_nums.const_4.expr()))
                * const_nums.const_2.expr(),
        );

        builder.store(
            &iv.get_at(dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_iv_reads),
        );

        let compress_iv_values =
            builder.constant_array::<U64Register>(&COMPRESS_IV.map(u64_to_le_field_bytes));
        let compress_iv = builder.uninit_slice();
        for (i, value) in compress_iv_values.iter().enumerate() {
            builder.store(
                &compress_iv.get(i),
                value,
                &Time::zero(),
                Some(*num_compresses),
            );
        }

        // The dummy iv_compress value is read twice for all rows other than the first four rows
        // of each compress round.
        builder.watch(num_rows, "num_rows");
        builder.watch(num_compresses, "num_compresses");
        let num_dummy_iv_compress_reads = builder.alloc_public::<ElementRegister>();
        builder.set_to_expression(
            &num_dummy_iv_compress_reads,
            (num_rows.expr() - (num_compresses.expr() * const_nums.const_4.expr()))
                * const_nums.const_2.expr(),
        );
        builder.store(
            &compress_iv.get_at(dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_iv_compress_reads),
        );

        let num_mix_iterations_element =
            builder.constant(&L::Field::from_canonical_usize(num_mix_iterations));

        // Each element is loaded once per compress cycle.
        let mut v_indices = MemoryArray::<L, 8, 4>::new(builder);
        for (i, indices) in V_INDICES.iter().enumerate() {
            v_indices.store_row(builder, i, indices, num_mix_iterations_element);
        }

        let mut v_last_write_ages = MemoryArray::<L, 8, 4>::new(builder);
        for (i, ages) in V_LAST_WRITE_AGES.iter().enumerate() {
            v_last_write_ages.store_row(builder, i, ages, num_mix_iterations_element);
        }

        let num_compresses_plus_1 = builder.alloc_public::<ElementRegister>();
        builder.set_to_expression(
            &num_compresses_plus_1,
            num_compresses.expr() + const_nums.const_1.expr(),
        );
        let mut permutations = MemoryArray::<L, 12, 16>::new(builder);
        for (i, permutation) in SIGMA_PERMUTATIONS.iter().enumerate() {
            permutations.store_row(
                builder,
                i,
                permutation,
                if i < num_mix_iterations_last_round {
                    num_compresses_plus_1
                } else {
                    *num_compresses
                },
            );
        }

        BLAKE2BConsts {
            iv,
            iv_values,
            compress_iv,
            v_indices,
            v_last_write_ages,
            permutations,
            dummy_index,
            dummy_ts,
            first_compress_h_read_ts,
        }
    }

    // This function will create all the registers/memory slots that will be used for control flow
    // related functions.
    #[allow(clippy::too_many_arguments)]
    pub fn blake2b_trace_data(
        builder: &mut BytesBuilder<L>,
        const_nums: &BLAKE2BConstNums,
        consts: &BLAKE2BConsts<L>,
        num_compresses: &ElementRegister,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        num_dummy_rounds: usize,
        length_last_round: usize,
        length_last_round_element: &ElementRegister,
    ) -> BLAKE2BTraceData {
        let (cycle_3_end_bit, cycle_4_end_bit, cycle_8_end_bit, cycle_96_end_bit) =
            Self::cycles_end_bits(builder);

        // Allocate end_bits from public input.
        let end_bit = builder.uninit_slice();
        for (i, end_bit_val) in end_bits.iter().enumerate() {
            builder.store(
                &end_bit.get(i),
                end_bit_val,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }

        let true_const = builder.constant::<BitRegister>(&L::Field::from_canonical_usize(1));
        let false_const = builder.constant::<BitRegister>(&L::Field::from_canonical_usize(0));

        for i in 0..num_dummy_rounds - 1 {
            builder.store(
                &end_bit.get(i + end_bits.len()),
                false_const,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }
        builder.store(
            &end_bit.get(num_dummy_rounds - 1 + end_bits.len()),
            false_const,
            &Time::zero(),
            Some(*length_last_round_element),
        );

        let digest_bit = builder.uninit_slice();
        for (i, digest_bit_val) in digest_bits.iter().enumerate() {
            builder.store(
                &digest_bit.get(i),
                digest_bit_val,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }
        for i in 0..num_dummy_rounds - 1 {
            builder.store(
                &digest_bit.get(i + digest_bits.len()),
                false_const,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }
        builder.store(
            &digest_bit.get(num_dummy_rounds - 1 + digest_bits.len()),
            false_const,
            &Time::zero(),
            Some(*length_last_round_element),
        );

        // `compress_id` is a register is computed by counting the number of cycles. We do this by
        // setting `process_id` to be the cumulative sum of the `end_bit` of each cycle.
        let compress_id: ElementRegister = builder.alloc::<ElementRegister>();
        builder.set_to_expression_first_row(&compress_id, L::Field::ZERO.into());
        builder.set_to_expression_transition(
            &compress_id.next(),
            compress_id.expr() + cycle_96_end_bit.expr(),
        );

        let mix_index = builder.alloc::<ElementRegister>();
        builder.set_to_expression_first_row(&mix_index, L::Field::ZERO.into());
        builder.set_to_expression_transition(
            &mix_index.next(),
            cycle_8_end_bit.not_expr() * (mix_index.expr() + const_nums.const_1.expr())
                + cycle_8_end_bit.expr() * const_nums.const_0.expr(),
        );

        // The array index register can be computed as `clock - process_id * CYCLE_LENGTH`.
        let clk = builder.clk;
        let compress_index =
            builder.expression(clk.expr() - compress_id.expr() * const_nums.const_96.expr());

        let compress_iteration = builder.alloc::<ElementRegister>();
        builder.set_to_expression_first_row(&compress_iteration, L::Field::ZERO.into());
        builder.set_to_expression_transition(
            &compress_iteration.next(),
            cycle_8_end_bit.not_expr() * compress_iteration.expr()
                + cycle_8_end_bit.expr()
                    * (cycle_96_end_bit.expr() * const_nums.const_0.expr()
                        + (cycle_96_end_bit.not_expr()
                            * (compress_iteration.expr() + const_nums.const_1.expr()))),
        );

        let at_end_compress = builder.load(&end_bit.get_at(compress_id), &Time::zero());
        let at_first_compress = builder.alloc::<BitRegister>();
        builder.set_to_expression_first_row(&at_first_compress, L::Field::ONE.into());
        builder.set_to_expression_transition(
            &at_first_compress.next(),
            (cycle_96_end_bit.not_expr() * at_first_compress.expr())
                + (cycle_96_end_bit.expr() * at_end_compress.expr()),
        );

        // Set previous compress id.  If we are the first compress, then set to
        // first_compress_h_read_ts.
        let mut previous_compress_id =
            builder.expression(compress_id.expr() - const_nums.const_1.expr());

        previous_compress_id = builder.select(
            at_first_compress,
            &consts.first_compress_h_read_ts,
            &previous_compress_id,
        );
        builder.watch(&previous_compress_id, "previous compress id");

        // Flag if we are within the first four rows of a compress.  In these rows, we will need to
        // use the COMPRESS_IV values.
        let is_compress_initialize = builder.alloc::<BitRegister>();
        builder.set_to_expression_first_row(&is_compress_initialize, L::Field::ONE.into());
        builder.set_to_expression_transition(
            &is_compress_initialize.next(),
            (cycle_96_end_bit.expr() * const_nums.const_1.expr())
                + (cycle_96_end_bit.not_expr()
                    * (cycle_4_end_bit.expr() * const_nums.const_0.expr()
                        + cycle_4_end_bit.not_expr() * is_compress_initialize.expr())),
        );

        // Flag if we are in the first row of a hash.  In that case, we will need to do an
        // xor for the v_12 value.
        let is_compress_first_row = builder.alloc::<BitRegister>();
        builder.set_to_expression_first_row(&is_compress_first_row, L::Field::ONE.into());
        builder
            .set_to_expression_transition(&is_compress_first_row.next(), cycle_96_end_bit.expr());

        // Flag if we are in the 3rd row of a hash.  In that case, we will need to do a xor on
        // the v_14 value.
        let is_compress_third_row =
            builder.expression(is_compress_initialize.expr() * cycle_3_end_bit.expr());

        // Need to flag to the last 4 rows of the compress cycle.
        // At those rows, the V values should be saved to v_final, so that those values can be used
        // to calculate the compress h values.
        let save_final_v: Slice<BitRegister> = builder.uninit_slice();
        let num_compresses_plus_1 = builder.alloc_public::<ElementRegister>();
        builder.set_to_expression(
            &num_compresses_plus_1,
            num_compresses.expr() + const_nums.const_1.expr(),
        );
        for i in 0..96 {
            builder.store(
                &save_final_v.get(i),
                if i < 92 { false_const } else { true_const },
                &Time::zero(),
                Some(if i < length_last_round {
                    num_compresses_plus_1
                } else {
                    *num_compresses
                }),
            );
        }
        let is_compress_finalize =
            builder.load(&save_final_v.get_at(compress_index), &Time::zero());
        builder.watch(&is_compress_finalize, "is_compress_finalize");

        let at_partial_compress_memory = builder.uninit_slice();
        for i in 0..end_bits.len() + num_dummy_rounds - 1 {
            builder.store(
                &at_partial_compress_memory.get(i),
                false_const,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }
        builder.store(
            &at_partial_compress_memory.get(end_bits.len() + num_dummy_rounds - 1),
            true_const,
            &Time::zero(),
            Some(*length_last_round_element),
        );
        let at_partial_compress = builder.load(
            &at_partial_compress_memory.get_at(compress_id),
            &Time::zero(),
        );
        builder.watch(&at_partial_compress, "at_partial_compress");

        // If we are the digest compress of the message, then save the digest.
        let at_digest_compress = builder.load(&digest_bit.get_at(compress_id), &Time::zero());
        let is_digest_row = builder.expression(cycle_96_end_bit.expr() * at_digest_compress.expr());
        builder.watch(&is_digest_row, "is_digest_row");

        BLAKE2BTraceData {
            clk,
            is_compress_initialize,
            is_compress_first_row,
            is_compress_third_row,
            is_digest_row,
            is_compress_finalize,
            at_first_compress,
            at_digest_compress,
            at_partial_compress,
            is_compress_final_row: cycle_96_end_bit,
            compress_id,
            previous_compress_id,
            compress_index,
            compress_iteration,
            mix_index,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn blake2b_memory(
        builder: &mut BytesBuilder<L>,
        padded_chunks: &[ArrayRegister<U64Register>],
        t_values: &ArrayRegister<U64Register>,
        const_nums: &BLAKE2BConstNums,
        consts: &BLAKE2BConsts<L>,
        num_messages: &ElementRegister,
        num_compresses: &ElementRegister,
        num_dummy_rounds: usize,
        length_last_round: usize,
    ) -> BLAKE2BMemory {
        let h = builder.uninit_slice();

        let num_messages_plus_1 = builder.alloc_public::<ElementRegister>();
        builder.set_to_expression(
            &num_messages_plus_1,
            num_messages.expr() + const_nums.const_1.expr(),
        );
        // Set dummy reads for h
        // Every row in the first compress of each message will read it 10 times. (96 * 10 * (num_messages + 1))
        // For the non first compress of each message
        //    1) First four rows will read it 8 times.  4 * 8
        //    2) Last row will read it 2 times.    2
        //    3) All other rows will read it 10 times.   91 * 10
        // For the partial compress, it will read it 10 times per row. (length_last_round * 10)
        let num_dummy_h_reads = builder.alloc_public::<ElementRegister>();
        let length_last_round_element =
            builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(length_last_round));
        let num_non_first_compresses = builder.alloc_public::<ElementRegister>();
        builder.set_to_expression(
            &num_non_first_compresses,
            num_compresses.expr() - num_messages_plus_1.expr(),
        );
        builder.set_to_expression(
            &num_dummy_h_reads,
            (num_messages_plus_1.expr() * const_nums.const_96.expr() * const_nums.const_10.expr())
                + (num_non_first_compresses.expr()
                    * (const_nums.const_4.expr() * const_nums.const_8.expr()
                        + const_nums.const_2.expr()
                        + const_nums.const_91.expr() * const_nums.const_10.expr()))
                + (length_last_round_element.expr() * const_nums.const_10.expr()),
        );
        builder.store(
            &h.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            Some(num_dummy_h_reads),
        );

        // Initialize the v memory
        // Need to set DUMMY_VALUE for all first four rows of every compress round.
        // It will be read 4 times per row.
        // Multiplicities = num_compress_round * 4 * 4;
        // It will also be read 4 times for every partial compress row.
        let num_dummy_v_reads =
            builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(
                (padded_chunks.len() + num_dummy_rounds - 1) * 4 * 4 + length_last_round * 4,
            ));
        let v = builder.uninit_slice();
        builder.store(
            &v.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            Some(num_dummy_v_reads),
        );

        // Initialize the v final memory
        let v_final = builder.uninit_slice();

        // Need to set dummy reads.
        // It will be num_compresses * (16 * 95) + length_last_round * 16
        let num_dummy_v_final_reads =
            builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(
                (padded_chunks.len() + num_dummy_rounds - 1) * (16 * 95) + length_last_round * 16,
            ));
        builder.store(
            &v_final.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            Some(num_dummy_v_final_reads),
        );

        // Initialize the m memory
        let m = builder.uninit_slice();

        // Each message chunk will be read 24 times per compress.  Two times per compress iteration.
        for (compress_id_value, padded_chunk) in padded_chunks.iter().enumerate() {
            assert!(padded_chunk.len() == 16);
            for (j, word) in padded_chunk.iter().enumerate() {
                builder.store(
                    &m.get(compress_id_value * 16 + j),
                    word,
                    &Time::zero(),
                    Some(const_nums.const_24),
                );
            }
        }
        for i in 0..num_dummy_rounds - 1 {
            for j in 0..16 {
                builder.store(
                    &m.get((i + padded_chunks.len()) * 16 + j),
                    const_nums.const_0_u64,
                    &Time::zero(),
                    Some(const_nums.const_24),
                );
            }
        }
        for i in 0..length_last_round {
            builder.store(
                &m.get((num_dummy_rounds - 1 + padded_chunks.len()) * 16 + i),
                const_nums.const_0_u64,
                &Time::zero(),
                Some(const_nums.const_24),
            );
        }

        let t = builder.uninit_slice();
        for (compress_id, t_value) in t_values.iter().enumerate() {
            builder.store(
                &t.get(compress_id),
                t_value,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }
        for i in 0..num_dummy_rounds {
            builder.store(
                &t.get(i + t_values.len()),
                const_nums.const_0_u64,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }

        BLAKE2BMemory {
            h,
            v,
            v_final,
            m,
            t,
        }
    }

    pub fn blake2b_data(
        builder: &mut BytesBuilder<L>,
        padded_chunks: &[ArrayRegister<U64Register>],
        t_values: &ArrayRegister<U64Register>,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: &ArrayRegister<ElementRegister>,
        num_messages: &ElementRegister,
    ) -> BLAKE2BData<L> {
        assert_eq!(padded_chunks.len(), end_bits.len());

        let num_real_rounds = padded_chunks.len();
        let degree_log = log2_ceil(num_real_rounds * 96);
        assert!(degree_log < 31, "AIR degree is too large");
        debug!("AIR degree after padding: {}", 1 << degree_log);
        let num_dummy_rounds = (1 << degree_log) / 96 + 1 - num_real_rounds;
        // Keep track of the last round length to know how many dummy reads to add.
        let length_last_round = (1 << degree_log) % 96;
        let num_rows =
            builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(1 << degree_log));

        // create the const numbers data
        let const_nums = Self::blake2b_const_nums(builder);

        // The number of full compress rounds
        let num_compresses = builder.constant(&L::Field::from_canonical_usize(
            num_real_rounds + num_dummy_rounds - 1,
        ));

        let length_last_round_element =
            builder.constant(&L::Field::from_canonical_usize(length_last_round));

        let mut num_mix_iterations = (num_real_rounds + num_dummy_rounds - 1) * 12;
        let num_mix_iterations_last_round = length_last_round / 8;
        assert!(num_mix_iterations_last_round == 4 || num_mix_iterations_last_round == 8);
        num_mix_iterations += num_mix_iterations_last_round;

        let public = BLAKE2BPublicData {
            padded_chunks: padded_chunks.to_vec(),
            t_values: *t_values,
            end_bits: *end_bits,
            digest_indices: *digest_indices,
        };

        // create the consts data
        let consts = Self::blake2b_const(
            builder,
            &num_rows,
            num_messages,
            &num_compresses,
            num_mix_iterations,
            num_mix_iterations_last_round,
            &const_nums,
        );

        // create the trace data
        let trace = Self::blake2b_trace_data(
            builder,
            &const_nums,
            &consts,
            &num_compresses,
            end_bits,
            digest_bits,
            num_dummy_rounds,
            length_last_round,
            &length_last_round_element,
        );

        // create the memory data
        let memory = Self::blake2b_memory(
            builder,
            padded_chunks,
            t_values,
            &const_nums,
            &consts,
            num_messages,
            &num_compresses,
            num_dummy_rounds,
            length_last_round,
        );

        BLAKE2BData {
            public,
            trace,
            memory,
            consts,
            const_nums,
        }
    }

    /// This function will retrieve the v values that will be inputted into the mix function
    pub fn blake2b_compress_initialize(
        builder: &mut BytesBuilder<L>,
        data: &BLAKE2BData<L>,
    ) -> ([ElementRegister; 4], [U64Register; 4]) {
        builder.watch(&data.trace.compress_index, "compress index");

        let init_idx_1 = data.trace.compress_index;
        let init_idx_2 = builder.add(data.trace.compress_index, data.const_nums.const_4);

        // Read the h values.
        //
        // Read the dummy index from h at any of the following conditions
        // 1) in the first compress of a message (first 4 rows will read from IV instead)
        // 2) NOT in the first 4 rows of compress (e.g. not is_compress_initialize)
        // 3) in the partial compress.
        //
        // Boolean expression is at_first_compress OR NOT(is_compress_initialize) OR at_partial_compress
        // That is equivalent to
        // NOT(NOT(at_first_compress) AND is_compress_initialize AND NOT(at_partial_compress))
        let read_dummy_h_idx = builder.alloc::<BitRegister>();
        builder.set_to_expression(
            &read_dummy_h_idx,
            data.const_nums.const_1.expr()
                - (data.trace.at_first_compress.not_expr()
                    * data.trace.is_compress_initialize.expr()
                    * data.trace.at_partial_compress.not_expr()),
        );
        builder.watch(&read_dummy_h_idx, "read dummy h idx");

        let h_idx_1 = builder.select(read_dummy_h_idx, &data.consts.dummy_index, &init_idx_1);
        let h_idx_2 = builder.select(read_dummy_h_idx, &data.consts.dummy_index, &init_idx_2);
        let h_ts = builder.select(
            read_dummy_h_idx,
            &data.consts.dummy_ts,
            &data.trace.previous_compress_id,
        );

        let mut h_value_1 = builder.load(&data.memory.h.get_at(h_idx_1), &Time::from_element(h_ts));
        let mut h_value_2 = builder.load(&data.memory.h.get_at(h_idx_2), &Time::from_element(h_ts));
        builder.watch(&h_value_1, "h_value_1");

        // Read the iv values.
        //
        // Read the dummy value at any of the following conditions
        // 1) NOT in the first 4 rows of compress (e.g. not is_compress_initialize)
        // 2) NOT the first compress of a messge
        // 3) In the partial compress
        //
        // Boolean expression is NOT(is_compress_initialize) OR NOT(at_first_compress) OR at_partial_compress
        // That is equivalent to
        // NOT(is_compress_initialize AND at_first_compress AND NOT(at_partial_compress))
        builder.watch(&data.trace.at_first_compress, "at_first_compress");
        let read_dummy_iv_idx = builder.alloc::<BitRegister>();
        builder.set_to_expression(
            &read_dummy_iv_idx,
            data.const_nums.const_1.expr()
                - (data.trace.is_compress_initialize.expr()
                    * data.trace.at_first_compress.expr()
                    * data.trace.at_partial_compress.not_expr()),
        );
        builder.watch(&read_dummy_iv_idx, "read_dummy_iv_idx");
        let iv_idx_1 = builder.select(read_dummy_iv_idx, &data.consts.dummy_index, &init_idx_1);
        let iv_idx_2 = builder.select(read_dummy_iv_idx, &data.consts.dummy_index, &init_idx_2);

        let iv_value_1 = builder.load(&data.consts.iv.get_at(iv_idx_1), &Time::zero());
        let iv_value_2 = builder.load(&data.consts.iv.get_at(iv_idx_2), &Time::zero());

        // Read the compress iv values.
        //
        // Read the dummy value at any of the following conditions
        // 1) NOT in the first 4 rows of compress (e.g. not is_compress_initialize)
        // 2) In the partial compress
        //
        // Boolean expression is NOT(is_compress_initialize) OR at_partial_compress
        // That is equivalent to
        // NOT(is_compress_initialize AND NOT(at_partial_compress))

        let read_dummy_compress_iv_idx = builder.alloc::<BitRegister>();
        builder.set_to_expression(
            &read_dummy_compress_iv_idx,
            data.const_nums.const_1.expr()
                - (data.trace.is_compress_initialize.expr()
                    * data.trace.at_partial_compress.not_expr()),
        );
        builder.watch(&read_dummy_compress_iv_idx, "read_dummy_compress_iv_idx");

        let compress_iv_idx_1 = builder.select(
            read_dummy_compress_iv_idx,
            &data.consts.dummy_index,
            &init_idx_1,
        );
        let compress_iv_idx_2 = builder.select(
            read_dummy_compress_iv_idx,
            &data.consts.dummy_index,
            &init_idx_2,
        );

        let compress_iv_value_1 = builder.load(
            &data.consts.compress_iv.get_at(compress_iv_idx_1),
            &Time::zero(),
        );
        let compress_iv_value_2 = builder.load(
            &data.consts.compress_iv.get_at(compress_iv_idx_2),
            &Time::zero(),
        );

        // Read the v values.
        //
        // First get the v indicies and last write timestamps.
        let v_indices = &data.consts.v_indices;
        let v1_idx = v_indices.get_at(builder, data.trace.mix_index, data.const_nums.const_0);
        let v2_idx = v_indices.get_at(builder, data.trace.mix_index, data.const_nums.const_1);
        let v3_idx = v_indices.get_at(builder, data.trace.mix_index, data.const_nums.const_2);
        let v4_idx = v_indices.get_at(builder, data.trace.mix_index, data.const_nums.const_3);

        let v_last_write_ages = &data.consts.v_last_write_ages;
        let v1_last_write_age =
            v_last_write_ages.get_at(builder, data.trace.mix_index, data.const_nums.const_0);
        let v2_last_write_age =
            v_last_write_ages.get_at(builder, data.trace.mix_index, data.const_nums.const_1);
        let v3_last_write_age =
            v_last_write_ages.get_at(builder, data.trace.mix_index, data.const_nums.const_2);
        let v4_last_write_age =
            v_last_write_ages.get_at(builder, data.trace.mix_index, data.const_nums.const_3);

        let mut v1_last_write_ts =
            builder.expression(data.trace.clk.expr() - v1_last_write_age.expr());
        let mut v2_last_write_ts =
            builder.expression(data.trace.clk.expr() - v2_last_write_age.expr());
        let mut v3_last_write_ts =
            builder.expression(data.trace.clk.expr() - v3_last_write_age.expr());
        let mut v4_last_write_ts =
            builder.expression(data.trace.clk.expr() - v4_last_write_age.expr());

        // Read the dummy value at any of the following conditions
        // 1) In the first 4 rows of compress (e.g. not is_compress_initialize)
        // 2) In the partial compress
        let read_dummy_v_idx = builder.or(
            data.trace.is_compress_initialize,
            data.trace.at_partial_compress,
        );
        builder.watch(&read_dummy_v_idx, "read_dummy_v_idx");

        v1_last_write_ts =
            builder.select(read_dummy_v_idx, &data.consts.dummy_ts, &v1_last_write_ts);

        v2_last_write_ts =
            builder.select(read_dummy_v_idx, &data.consts.dummy_ts, &v2_last_write_ts);

        v3_last_write_ts =
            builder.select(read_dummy_v_idx, &data.consts.dummy_ts, &v3_last_write_ts);

        v4_last_write_ts =
            builder.select(read_dummy_v_idx, &data.consts.dummy_ts, &v4_last_write_ts);

        let v1_read_idx = builder.select(read_dummy_v_idx, &data.consts.dummy_index, &v1_idx);
        let v2_read_idx = builder.select(read_dummy_v_idx, &data.consts.dummy_index, &v2_idx);
        let v3_read_idx = builder.select(read_dummy_v_idx, &data.consts.dummy_index, &v3_idx);
        let v4_read_idx = builder.select(read_dummy_v_idx, &data.consts.dummy_index, &v4_idx);

        let mut v1_value = builder.load(
            &data.memory.v.get_at(v1_read_idx),
            &Time::from_element(v1_last_write_ts),
        );
        let mut v2_value = builder.load(
            &data.memory.v.get_at(v2_read_idx),
            &Time::from_element(v2_last_write_ts),
        );
        let mut v3_value = builder.load(
            &data.memory.v.get_at(v3_read_idx),
            &Time::from_element(v3_last_write_ts),
        );
        let mut v4_value = builder.load(
            &data.memory.v.get_at(v4_read_idx),
            &Time::from_element(v4_last_write_ts),
        );

        // Set the v values based on where in the compress we are.

        // Set v1 and v2 value.
        // Use the iv values if we are in the first 4 rows of a message.
        let use_iv_values = builder.and(
            data.trace.is_compress_initialize,
            data.trace.at_first_compress,
        );

        h_value_1 = builder.select(use_iv_values, &iv_value_1, &h_value_1);
        h_value_2 = builder.select(use_iv_values, &iv_value_2, &h_value_2);

        // If we are in the first 4 rows of a compress, then we will need to use the h values, else use the v values.
        v1_value = builder.select(data.trace.is_compress_initialize, &h_value_1, &v1_value);
        v2_value = builder.select(data.trace.is_compress_initialize, &h_value_2, &v2_value);

        // Set v3 and v4 value.
        // Use the compress iv values if we are in the first 4 rows of a compress, else use the v values
        v3_value = builder.select(
            data.trace.is_compress_initialize,
            &compress_iv_value_1,
            &v3_value,
        );
        v4_value = builder.select(
            data.trace.is_compress_initialize,
            &compress_iv_value_2,
            &v4_value,
        );

        // If we are at the first compress row, then will need to xor v4 with t
        let t = builder.load(&data.memory.t.get_at(data.trace.compress_id), &Time::zero());

        let v4_xor_t = builder.xor(v4_value, t);
        v4_value = builder.select(data.trace.is_compress_first_row, &v4_xor_t, &v4_value);

        // If we are at the third compress row, then will need to xor v4 with 0xFFFFFFFFFFFFFFFF
        let inverse_v4_value = builder.xor(&v4_value, &data.const_nums.const_ffffffffffffffff);
        let use_inverse_v4_value = builder.mul(
            data.trace.at_digest_compress,
            data.trace.is_compress_third_row,
        );
        v4_value = builder.select(use_inverse_v4_value, &inverse_v4_value, &v4_value);

        (
            [v1_idx, v2_idx, v3_idx, v4_idx],
            [v1_value, v2_value, v3_value, v4_value],
        )
    }

    /// The processing step of a BLAKE2B round.
    pub fn blake2b_compress(
        builder: &mut BytesBuilder<L>,
        v_indices: &[ElementRegister; 4],
        v_values: &[U64Register; 4],
        data: &BLAKE2BData<L>,
    ) -> Vec<ArrayRegister<U64Register>> {
        let num_digests = data.public.digest_indices.len();

        // Create the public registers to verify the hash.
        let hash_state_public = (0..num_digests)
            .map(|_| builder.alloc_array_public(4))
            .collect::<Vec<_>>();

        /*
        let state_ptr = builder.uninit_slice();

        for (i, h_slice) in data
            .public
            .digest_indices
            .iter()
            .zip(hash_state_public.iter())
        {
            for (j, h) in h_slice.iter().enumerate() {
                builder.free(&state_ptr.get(j), h, &Time::from_element(i));
            }
        }
        */

        // Load the permutation values.
        let mut permutation_col: ElementRegister =
            builder.mul(data.trace.mix_index, data.const_nums.const_2);

        let m_idx_1 = data.consts.permutations.get_at(
            builder,
            data.trace.compress_iteration,
            permutation_col,
        );
        builder.set_to_expression(
            &m_idx_1,
            data.trace.compress_id.expr() * data.const_nums.const_16.expr() + m_idx_1.expr(),
        );
        permutation_col = builder.add(permutation_col, data.const_nums.const_1);

        let m_idx_2 = data.consts.permutations.get_at(
            builder,
            data.trace.compress_iteration,
            permutation_col,
        );
        builder.set_to_expression(
            &m_idx_2,
            data.trace.compress_id.expr() * data.const_nums.const_16.expr() + m_idx_2.expr(),
        );

        // Load the message values.
        let m_1 = builder.load(&data.memory.m.get_at(m_idx_1), &Time::zero());
        let m_2 = builder.load(&data.memory.m.get_at(m_idx_2), &Time::zero());

        // Output the "parameters" being sent to the mix function.
        builder.watch(&v_values[0], "va");
        builder.watch(&v_values[1], "vb");
        builder.watch(&v_values[2], "vc");
        builder.watch(&v_values[3], "vd");
        builder.watch(&m_1, "m_1");
        builder.watch(&m_2, "m_2");

        let (updated_v0, updated_v1, updated_v2, updated_v3) = Self::blake2b_mix(
            builder,
            &v_values[0],
            &v_values[1],
            &v_values[2],
            &v_values[3],
            &m_1,
            &m_2,
        );

        // Save the output of the mix in v or v_final.

        // Save the output into v in all of the following conditions.
        // 1) NOT in the last 4 rows of compress (e.g. not is_compress_finalize)
        // 2) NOT in the partial compress.
        //
        // Boolean expression is NOT(is_compress_initialize) AND NOT(at_partial_compress)
        let save_v = builder.alloc::<BitRegister>();
        builder.set_to_expression(
            &save_v,
            data.trace.is_compress_finalize.not_expr() * data.trace.at_partial_compress.not_expr(),
        );
        builder.watch(&save_v, "save_v");

        // Save the output into v in all of the following conditions.
        // 1) in the last 4 rows of compress (e.g. is_compress_finalize)
        // 2) NOT in the partial compress.
        //
        // Boolean expression is is_compress_finalize AND NOT(at_partial_compress)
        let save_v_final = builder.alloc::<BitRegister>();
        builder.set_to_expression(
            &save_v_final,
            data.trace.is_compress_finalize.expr() * data.trace.at_partial_compress.not_expr(),
        );
        builder.watch(&save_v_final, "save_v_final");

        let updated_v_values = [updated_v0, updated_v1, updated_v2, updated_v3];
        for (value, v_index) in updated_v_values.iter().zip(v_indices.iter()) {
            builder.store(
                &data.memory.v.get_at(*v_index),
                *value,
                &Time::from_element(builder.clk),
                Some(save_v.as_element()),
            );

            builder.store(
                &data.memory.v_final.get_at(*v_index),
                *value,
                &Time::from_element(data.trace.compress_id),
                Some(save_v_final.as_element()),
            );
        }

        // If we are at the last row of compress, then compute and save the h value.

        // First load the previous round's h value.
        let h_workspace_1 = builder.alloc_array::<U64Register>(8);

        // Read dummy h values if any of the following conditions are true
        // 1) NOT at last row of a compress
        // 2) at the first compress
        //
        // Boolean expression is NOT(is_compress_final_row) OR at_first_compress
        // That is equivalent to
        // NOT(is_compress_final_row AND NOT(at_first_compress))
        let read_dummy_h_idx = builder.alloc::<BitRegister>();
        builder.set_to_expression(
            &read_dummy_h_idx,
            data.const_nums.const_1.expr()
                - (data.trace.is_compress_final_row.expr()
                    * data.trace.at_first_compress.not_expr()),
        );

        builder.watch(&read_dummy_h_idx, "read_dummy_h_idx 2");
        let h_ts = builder.select(
            read_dummy_h_idx,
            &data.consts.dummy_ts,
            &data.trace.previous_compress_id,
        );
        for i in 0..8 {
            let i_element = builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(i));
            let h_idx = builder.select(read_dummy_h_idx, &data.consts.dummy_index, &i_element);
            let mut h_value = builder.load(&data.memory.h.get_at(h_idx), &Time::from_element(h_ts));
            builder.watch(&h_value, "h_value 2");

            // If we are at the first compress of a message, then use the iv values instead of the h values.
            h_value = builder.select(
                data.trace.at_first_compress,
                &data.consts.iv_values.get(i),
                &h_value,
            );
            builder.set_to_expression(&h_workspace_1.get(i), h_value.expr());
        }

        // Xor the first 8 final v values
        let h_workspace_2 = builder.alloc_array::<U64Register>(8);

        // Read dummy v_final values if NOT at last row of a compress
        let read_dummy_v_final_idx = builder.not(data.trace.is_compress_final_row);
        builder.watch(&read_dummy_v_final_idx, "read_dummy_v_final_idx");
        let v_ts = builder.select(
            read_dummy_v_final_idx,
            &data.trace.compress_id,
            &data.consts.dummy_ts,
        );
        for i in 0..8 {
            let i_element = builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(i));
            let v_final_idx =
                builder.select(read_dummy_v_final_idx, &data.consts.dummy_index, &i_element);
            let v_i = builder.load(
                &data.memory.v_final.get_at(v_final_idx),
                &Time::from_element(v_ts),
            );
            let updated_h = builder.xor(h_workspace_1.get(i), v_i);
            builder.set_to_expression(&h_workspace_2.get(i), updated_h.expr());
        }

        // Xor the second 8 final v values
        let h = builder.alloc_array::<U64Register>(8);
        for i in 0..8 {
            let i_element = builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(i));
            let i_element_plus_8 = builder.add(i_element, data.const_nums.const_8);

            let v_final_idx = builder.select(
                read_dummy_v_final_idx,
                &data.consts.dummy_index,
                &i_element_plus_8,
            );

            let v_value = builder.load(
                &data.memory.v_final.get_at(v_final_idx),
                &Time::from_element(v_ts),
            );
            let xor = builder.xor(h_workspace_2.get(i), v_value);
            builder.set_to_expression(&h.get(i), xor.expr());
            builder.watch(&xor, "final h");

            let h_idx = builder.select(
                data.trace.is_compress_final_row,
                &i_element,
                &data.consts.dummy_index,
            );

            // Need to save the h value twice as it will be written twice per compress.
            let h_multiplicity = builder.select(
                data.trace.is_compress_final_row,
                &data.const_nums.const_2,
                &data.const_nums.const_0,
            );
            builder.store(
                &data.memory.h.get_at(h_idx),
                xor,
                &Time::from_element(data.trace.compress_id),
                Some(h_multiplicity),
            );
        }

        /*
        for (i, element) in h.get_subarray(0..4).iter().enumerate() {
            builder.store(
                &state_ptr.get(i),
                element,
                &Time::from_element(data.trace.compress_id),
                Some(data.trace.is_digest_row.as_element()),
            );
        }
        */

        builder.watch_memory(&data.consts.iv.get(0), "iv[0]");
        builder.watch_memory(&data.consts.iv.get(7), "iv[7]");
        builder.watch_memory(
            &data.consts.iv.get_at(data.consts.dummy_index),
            "iv[dummy_index]",
        );
        builder.watch_memory(&data.consts.compress_iv.get(0), "compress_iv[0]");
        builder.watch_memory(&data.consts.compress_iv.get(7), "compress_iv[7]");
        builder.watch_memory(
            &data.consts.compress_iv.get_at(data.consts.dummy_index),
            "compress_iv[dummy_index]",
        );
        builder.watch_memory(
            &data.consts.v_indices.flattened_memory.get(0),
            "v_indices[0]",
        );
        builder.watch_memory(
            &data.consts.v_indices.flattened_memory.get(31),
            "v_indices[31]",
        );
        builder.watch_memory(
            &data.consts.v_last_write_ages.flattened_memory.get(0),
            "v_last_write_ages[0]",
        );
        builder.watch_memory(
            &data.consts.v_last_write_ages.flattened_memory.get(31),
            "v_last_write_ages[31]",
        );
        builder.watch_memory(
            &data.consts.permutations.flattened_memory.get(0),
            "permutations[0]",
        );
        builder.watch_memory(
            &data.consts.permutations.flattened_memory.get(127),
            "permutations[127]",
        );
        //builder.watch_memory(&data.memory.h.get(0), "h[0]");
        //builder.watch_memory(&data.memory.h.get(7), "h[7]");
        builder.watch_memory(
            &data.memory.h.get_at(data.consts.dummy_index),
            "h[dummy_index]",
        );
        builder.watch_memory(&data.memory.v.get(0), "v[0]");
        //builder.watch_memory(&data.memory.v.get(15), "v[15]");
        builder.watch_memory(
            &data.memory.v.get_at(data.consts.dummy_index),
            "v[dummy_index]",
        );
        builder.watch_memory(&data.memory.v_final.get(0), "v_final[0]");
        builder.watch_memory(
            &data.memory.v_final.get_at(data.consts.dummy_index),
            "v_final[dummy_index]",
        );

        hash_state_public
    }

    pub fn blake2b_mix(
        builder: &mut BytesBuilder<L>,
        v_a: &U64Register,
        v_b: &U64Register,
        v_c: &U64Register,
        v_d: &U64Register,
        x: &U64Register,
        y: &U64Register,
    ) -> (U64Register, U64Register, U64Register, U64Register) {
        let mut v_a_inter = builder.add(*v_a, *v_b);
        v_a_inter = builder.add(v_a_inter, *x);

        builder.watch(&v_a_inter, "after first operation, va");

        let mut v_d_inter = builder.xor(*v_d, v_a_inter);
        builder.watch(&v_d_inter, "after first xor, vd");
        v_d_inter = builder.rotate_right(v_d_inter, 32);
        builder.watch(&v_d_inter, "after first operation, vd");

        let mut v_c_inter = builder.add(*v_c, v_d_inter);
        builder.watch(&v_c_inter, "after first operation, vc");

        let mut v_b_inter = builder.xor(*v_b, v_c_inter);
        v_b_inter = builder.rotate_right(v_b_inter, 24);
        builder.watch(&v_b_inter, "after first operation, vb");

        v_a_inter = builder.add(v_a_inter, v_b_inter);
        v_a_inter = builder.add(v_a_inter, *y);
        builder.watch(&v_a_inter, "after second operation, va");

        v_d_inter = builder.xor(v_d_inter, v_a_inter);
        v_d_inter = builder.rotate_right(v_d_inter, 16);

        v_c_inter = builder.add(v_c_inter, v_d_inter);

        v_b_inter = builder.xor(v_b_inter, v_c_inter);
        v_b_inter = builder.rotate_right(v_b_inter, 63);

        (v_a_inter, v_b_inter, v_c_inter, v_d_inter)
    }
}
