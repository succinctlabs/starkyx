use log::debug;
use plonky2::util::log2_ceil;

use super::data::{BLAKE2BConstNums, BLAKE2BConsts, BLAKE2BData};
use super::register::BLAKE2BDigestRegister;
use super::{BLAKE2B, COMPRESS_LENGTH, IV, STATE_SIZE};
use crate::chip::memory::instruction::MemorySliceIndex;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::{u64_from_le_field_bytes, u64_to_le_field_bytes};
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::machine::hash::blake::blake2b::data::{
    BLAKE2BMemory, BLAKE2BPublicData, BLAKE2BTraceData, MemoryArray,
};
use crate::machine::hash::blake::blake2b::{
    COMPRESS_IV, MIX_LENGTH, MSG_ARRAY_SIZE, NUM_MIX_ROUNDS, SIGMA_PERMUTATIONS, V_INDICES,
    V_LAST_WRITE_AGES,
};
use crate::machine::hash::{HashDigest, HashIntConversion, HashInteger};
use crate::math::prelude::*;

impl<B: Builder> HashInteger<B> for BLAKE2B {
    type Value = <U64Register as Register>::Value<B::Field>;
    type IntRegister = U64Register;
}

impl<B: Builder> HashIntConversion<B> for BLAKE2B {
    fn int_to_field_value(int: Self::Integer) -> Self::Value {
        u64_to_le_field_bytes(int)
    }

    fn field_value_to_int(value: &Self::Value) -> Self::Integer {
        u64_from_le_field_bytes(value)
    }
}

impl<B: Builder> HashDigest<B> for BLAKE2B {
    type DigestRegister = BLAKE2BDigestRegister;
}

const DUMMY_INDEX: u64 = i32::MAX as u64;
const DUMMY_INDEX_2: u64 = (i32::MAX - 1) as u64;
const DUMMY_TS: u64 = (i32::MAX - 1) as u64;
const FIRST_COMPRESS_H_READ_TS: u64 = i32::MAX as u64;

pub trait BLAKEAir<B: Builder>: HashIntConversion<B> + HashDigest<B> {
    fn cycles_end_bits(builder: &mut B) -> (BitRegister, BitRegister, BitRegister, BitRegister);

    fn blake2b(
        builder: &mut B,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        t_values: &ArrayRegister<Self::IntRegister>,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: &ArrayRegister<ElementRegister>,
        num_messages: &ElementRegister,
    ) -> Vec<Self::DigestRegister>;

    fn blake2b_const_nums(builder: &mut B) -> BLAKE2BConstNums;

    #[allow(clippy::too_many_arguments)]
    fn blake2b_const(
        builder: &mut B,
        num_rows_element: &ElementRegister,
        num_messages_element: &ElementRegister,
        num_real_compresses: usize,
        num_real_compresses_element: &ElementRegister,
        num_dummy_compresses: usize,
        num_total_mix_iterations: usize,
        num_mix_iterations_last_compress: usize,
        const_nums: &BLAKE2BConstNums,
    ) -> BLAKE2BConsts<B>;

    #[allow(clippy::too_many_arguments)]
    fn blake2b_trace_data(
        builder: &mut B,
        const_nums: &BLAKE2BConstNums,
        consts: &BLAKE2BConsts<B>,
        num_real_compresses: usize,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        num_dummy_compresses: usize,
        length_last_compress: usize,
        length_last_compress_element: &ElementRegister,
    ) -> BLAKE2BTraceData;

    #[allow(clippy::too_many_arguments)]
    fn blake2b_memory(
        builder: &mut B,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        t_values: &ArrayRegister<Self::IntRegister>,
        const_nums: &BLAKE2BConstNums,
        consts: &BLAKE2BConsts<B>,
        num_messages_element: &ElementRegister,
        num_real_compresses: usize,
        num_real_compresses_element: &ElementRegister,
        num_dummy_rows: usize,
    ) -> BLAKE2BMemory;

    fn blake2b_data(
        builder: &mut B,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        t_values: &ArrayRegister<Self::IntRegister>,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: &ArrayRegister<ElementRegister>,
        num_messages_element: &ElementRegister,
    ) -> BLAKE2BData<B>;

    fn blake2b_compress_initialize(
        builder: &mut B,
        data: &BLAKE2BData<B>,
    ) -> ([ElementRegister; 4], [Self::IntRegister; 4]);

    fn blake2b_compress(
        builder: &mut B,
        v_indices: &[ElementRegister; 4],
        v_values: &[Self::IntRegister; 4],
        data: &BLAKE2BData<B>,
    );

    fn blake2b_compress_finalize(
        builder: &mut B,
        state_ptr: &Slice<Self::IntRegister>,
        data: &BLAKE2BData<B>,
    );

    fn blake2b_mix(
        builder: &mut B,
        v_a: &Self::IntRegister,
        v_b: &Self::IntRegister,
        v_c: &Self::IntRegister,
        v_d: &Self::IntRegister,
        x: &Self::IntRegister,
        y: &Self::IntRegister,
    ) -> (
        Self::IntRegister,
        Self::IntRegister,
        Self::IntRegister,
        Self::IntRegister,
    );
}

impl<L: AirParameters> BLAKEAir<BytesBuilder<L>> for BLAKE2B
where
    L::Instruction: UintInstructions,
{
    fn cycles_end_bits(
        builder: &mut BytesBuilder<L>,
    ) -> (BitRegister, BitRegister, BitRegister, BitRegister) {
        let cycle_4 = builder.cycle(2);
        let cycle_8 = builder.cycle(3);
        let loop_3 = builder.api().loop_instr(3);
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

    fn blake2b(
        builder: &mut BytesBuilder<L>,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        t_values: &ArrayRegister<Self::IntRegister>,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: &ArrayRegister<ElementRegister>,
        num_messages: &ElementRegister,
    ) -> Vec<Self::DigestRegister> {
        let data = Self::blake2b_data(
            builder,
            padded_chunks,
            t_values,
            end_bits,
            digest_bits,
            digest_indices,
            num_messages,
        );

        let state_ptr = builder.uninit_slice();
        let num_digests = data.public.digest_indices.len();

        // Create the public registers to input the expected digests.
        let hash_state_public_tmp: Vec<ArrayRegister<Self::IntRegister>> = (0..num_digests)
            .map(|_| builder.alloc_array_public::<Self::IntRegister>(4))
            .collect::<_>();

        let mut hash_state_public: Vec<Self::DigestRegister> = Vec::new();
        for i in hash_state_public_tmp.iter() {
            hash_state_public.push(Self::DigestRegister::from_array(*i));
        }

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

        let (v_indices, v_values) = Self::blake2b_compress_initialize(builder, &data);
        Self::blake2b_compress(builder, &v_indices, &v_values, &data);
        Self::blake2b_compress_finalize(builder, &state_ptr, &data);

        hash_state_public
    }

    fn blake2b_const_nums(builder: &mut BytesBuilder<L>) -> BLAKE2BConstNums {
        BLAKE2BConstNums {
            const_0: builder.constant(&L::Field::from_canonical_u8(0)),
            const_0_u64: builder
                .constant(&<Self as HashIntConversion<BytesBuilder<L>>>::int_to_field_value(0u64)),
            const_1: builder.constant(&L::Field::from_canonical_u8(1)),
            const_2: builder.constant(&L::Field::from_canonical_u8(2)),
            const_3: builder.constant(&L::Field::from_canonical_u8(3)),
            const_4: builder.constant(&L::Field::from_canonical_u8(4)),
            const_8: builder.constant(&L::Field::from_canonical_u8(8)),
            const_10: builder.constant(&L::Field::from_canonical_u8(10)),
            const_12: builder.constant(&L::Field::from_canonical_u8(12)),
            const_16: builder.constant(&L::Field::from_canonical_u8(16)),
            const_91: builder.constant(&L::Field::from_canonical_u8(91)),
            const_96: builder.constant(&L::Field::from_canonical_u8(96)),
            const_ffffffffffffffff: builder.constant::<Self::IntRegister>(
                &<Self as HashIntConversion<BytesBuilder<L>>>::int_to_field_value(
                    0xFFFFFFFFFFFFFFFF,
                ),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn blake2b_const(
        builder: &mut BytesBuilder<L>,
        num_rows_element: &ElementRegister,
        num_messages_element: &ElementRegister,
        num_real_compresses: usize,
        num_real_compresses_element: &ElementRegister,
        num_dummy_compresses: usize,
        num_total_mix_iterations: usize,
        num_mix_iterations_last_compress: usize,
        const_nums: &BLAKE2BConstNums,
    ) -> BLAKE2BConsts<BytesBuilder<L>> {
        assert!(DUMMY_INDEX < L::Field::order());
        let dummy_index: ElementRegister =
            builder.constant(&L::Field::from_canonical_u64(DUMMY_INDEX));

        let dummy_index_2: ElementRegister =
            builder.constant(&L::Field::from_canonical_u64(DUMMY_INDEX_2));

        assert!(DUMMY_TS < L::Field::order());
        let dummy_ts: ElementRegister = builder.constant(&L::Field::from_canonical_u64(DUMMY_TS));

        assert!(FIRST_COMPRESS_H_READ_TS < L::Field::order());
        let first_compress_h_read_ts: ElementRegister =
            builder.constant(&L::Field::from_canonical_u64(FIRST_COMPRESS_H_READ_TS));

        let iv_values = builder.constant_array::<Self::IntRegister>(
            &IV.map(&<Self as HashIntConversion<BytesBuilder<L>>>::int_to_field_value),
        );
        let iv: Slice<crate::chip::uint::register::ByteArrayRegister<8>> = builder.uninit_slice();
        for (i, value) in iv_values.iter().enumerate() {
            builder.store(
                &iv.get(i),
                value,
                &Time::zero(),
                Some(*num_messages_element),
                Some("iv".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        // The dummy iv value is read twice at the rows other than the first 4 rows of the each messages's
        // first compress round.
        let num_dummy_iv_reads = builder.public_expression(
            (num_rows_element.expr() - (num_messages_element.expr() * const_nums.const_4.expr()))
                * const_nums.const_2.expr(),
        );

        builder.store(
            &iv.get_at(dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_iv_reads),
            Some("iv".to_string()),
            Some(MemorySliceIndex::IndexElement(dummy_index)),
        );

        let compress_iv_values = builder.constant_array::<Self::IntRegister>(
            &COMPRESS_IV.map(&<Self as HashIntConversion<BytesBuilder<L>>>::int_to_field_value),
        );
        let compress_iv = builder.uninit_slice();
        for (i, value) in compress_iv_values.iter().enumerate() {
            builder.store(
                &compress_iv.get(i),
                value,
                &Time::zero(),
                Some(*num_real_compresses_element),
                Some("compress_iv".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }

        // The dummy iv_compress value is read twice for all rows other than the first four rows
        // of each real compress round.
        let num_dummy_iv_compress_reads = builder.public_expression(
            (num_rows_element.expr()
                - (num_real_compresses_element.expr() * const_nums.const_4.expr()))
                * const_nums.const_2.expr(),
        );
        builder.store(
            &compress_iv.get_at(dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_iv_compress_reads),
            Some("compress_iv".to_string()),
            Some(MemorySliceIndex::IndexElement(dummy_index)),
        );

        let num_total_mix_iterations_element = builder
            .constant::<ElementRegister>(&L::Field::from_canonical_usize(num_total_mix_iterations));
        let mut v_indices = MemoryArray::<BytesBuilder<L>, MIX_LENGTH, 4>::new(builder);
        for (i, indices) in V_INDICES.iter().enumerate() {
            v_indices.store_row(
                builder,
                i,
                indices,
                num_total_mix_iterations_element,
                Some("v_indices".to_string()),
            );
        }

        let mut v_last_write_ages = MemoryArray::<BytesBuilder<L>, MIX_LENGTH, 4>::new(builder);
        for (i, ages) in V_LAST_WRITE_AGES.iter().enumerate() {
            v_last_write_ages.store_row(
                builder,
                i,
                ages,
                num_total_mix_iterations_element,
                Some("v_last_write".to_string()),
            );
        }

        let mut permutations =
            MemoryArray::<BytesBuilder<L>, NUM_MIX_ROUNDS, MSG_ARRAY_SIZE>::new(builder);
        let num_compresses_element = builder.constant::<ElementRegister>(
            &L::Field::from_canonical_usize(num_real_compresses + num_dummy_compresses),
        );
        let num_full_compresses_element = builder.constant::<ElementRegister>(
            &L::Field::from_canonical_usize(num_real_compresses + num_dummy_compresses - 1),
        );

        for (i, permutation) in SIGMA_PERMUTATIONS.iter().enumerate() {
            permutations.store_row(
                builder,
                i,
                permutation,
                if i < num_mix_iterations_last_compress {
                    num_compresses_element
                } else {
                    num_full_compresses_element
                },
                Some("permutation".to_string()),
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
            dummy_index_2,
            dummy_ts,
            first_compress_h_read_ts,
        }
    }

    // This function will create all the registers/memory slots that will be used for control flow
    // related functions.
    #[allow(clippy::too_many_arguments)]
    fn blake2b_trace_data(
        builder: &mut BytesBuilder<L>,
        const_nums: &BLAKE2BConstNums,
        consts: &BLAKE2BConsts<BytesBuilder<L>>,
        num_real_compresses: usize,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        num_dummy_compresses: usize,
        length_last_compress: usize,
        length_last_compress_element: &ElementRegister,
    ) -> BLAKE2BTraceData {
        let (cycle_3_end_bit, cycle_4_end_bit, cycle_8_end_bit, cycle_96_end_bit) =
            Self::cycles_end_bits(builder);

        let true_const = builder.constant::<BitRegister>(&L::Field::from_canonical_usize(1));
        let false_const = builder.constant::<BitRegister>(&L::Field::from_canonical_usize(0));

        let num_total_compresses = num_real_compresses + num_dummy_compresses;

        // Allocate end_bits from public input.
        let end_bit = builder.uninit_slice();
        for (i, end_bit_val) in end_bits.iter().enumerate() {
            builder.store(
                &end_bit.get(i),
                end_bit_val,
                &Time::zero(),
                Some(const_nums.const_96),
                Some("end_bit".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        for i in num_real_compresses..num_total_compresses - 1 {
            builder.store(
                &end_bit.get(i),
                false_const,
                &Time::zero(),
                Some(const_nums.const_96),
                Some("end_bit".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        let last_compress_idx = num_total_compresses - 1;
        builder.store(
            &end_bit.get(last_compress_idx),
            false_const,
            &Time::zero(),
            Some(*length_last_compress_element),
            Some("end_bit".to_string()),
            Some(MemorySliceIndex::Index(last_compress_idx)),
        );

        let digest_bit = builder.uninit_slice();
        for (i, digest_bit_val) in digest_bits.iter().enumerate() {
            builder.store(
                &digest_bit.get(i),
                digest_bit_val,
                &Time::zero(),
                Some(const_nums.const_96),
                Some("digest_bit".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        for i in num_real_compresses..num_total_compresses - 1 {
            builder.store(
                &digest_bit.get(i),
                false_const,
                &Time::zero(),
                Some(const_nums.const_96),
                Some("digest_bit".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        builder.store(
            &digest_bit.get(last_compress_idx),
            false_const,
            &Time::zero(),
            Some(*length_last_compress_element),
            Some("digest_bit".to_string()),
            Some(MemorySliceIndex::Index(last_compress_idx)),
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

        let mix_id = builder.alloc::<ElementRegister>();
        builder.set_to_expression_first_row(&mix_id, L::Field::ZERO.into());
        builder.set_to_expression_transition(
            &mix_id.next(),
            cycle_8_end_bit.not_expr() * mix_id.expr()
                + cycle_8_end_bit.expr()
                    * (cycle_96_end_bit.expr() * const_nums.const_0.expr()
                        + (cycle_96_end_bit.not_expr()
                            * (mix_id.expr() + const_nums.const_1.expr()))),
        );

        let at_end_compress = builder.load(
            &end_bit.get_at(compress_id),
            &Time::zero(),
            Some("end_bit".to_string()),
            Some(MemorySliceIndex::IndexElement(compress_id)),
        );
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
        let num_compresses_element = builder.constant::<ElementRegister>(
            &L::Field::from_canonical_usize(num_real_compresses + num_dummy_compresses),
        );
        let num_full_compresses_element = builder.constant::<ElementRegister>(
            &L::Field::from_canonical_usize(num_real_compresses + num_dummy_compresses - 1),
        );
        for i in 0..COMPRESS_LENGTH {
            builder.store(
                &save_final_v.get(i),
                if i < 92 { false_const } else { true_const },
                &Time::zero(),
                Some(if i < length_last_compress {
                    num_compresses_element
                } else {
                    num_full_compresses_element
                }),
                Some("save_final_v".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        let is_compress_finalize = builder.load(
            &save_final_v.get_at(compress_index),
            &Time::zero(),
            Some("save_final_v".to_string()),
            Some(MemorySliceIndex::IndexElement(compress_index)),
        );

        let at_dummy_compress_memory = builder.uninit_slice();
        for i in 0..num_real_compresses {
            builder.store(
                &at_dummy_compress_memory.get(i),
                false_const,
                &Time::zero(),
                Some(const_nums.const_96),
                Some("at_dummy_compress_memory".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        for i in num_real_compresses..num_total_compresses - 1 {
            builder.store(
                &at_dummy_compress_memory.get(i),
                true_const,
                &Time::zero(),
                Some(const_nums.const_96),
                Some("at_dummy_compress_memory".to_string()),
                Some(MemorySliceIndex::Index(i)),
            );
        }
        builder.store(
            &at_dummy_compress_memory.get(last_compress_idx),
            true_const,
            &Time::zero(),
            Some(*length_last_compress_element),
            Some("at_dummy_compress_memory".to_string()),
            Some(MemorySliceIndex::Index(last_compress_idx)),
        );

        let at_dummy_compress = builder.load(
            &at_dummy_compress_memory.get_at(compress_id),
            &Time::zero(),
            Some("at_dummy_compress_memory".to_string()),
            Some(MemorySliceIndex::IndexElement(compress_id)),
        );

        // If we are the digest compress of the message, then save the digest.
        let at_digest_compress = builder.load(
            &digest_bit.get_at(compress_id),
            &Time::zero(),
            Some("digest_bit".to_string()),
            Some(MemorySliceIndex::IndexElement(compress_id)),
        );
        let is_digest_row = builder.expression(cycle_96_end_bit.expr() * at_digest_compress.expr());

        BLAKE2BTraceData {
            clk,
            is_compress_initialize,
            is_compress_first_row,
            is_compress_third_row,
            is_digest_row,
            is_compress_finalize,
            at_first_compress,
            at_digest_compress,
            at_end_compress,
            at_dummy_compress,
            is_compress_final_row: cycle_96_end_bit,
            compress_id,
            previous_compress_id,
            compress_index,
            mix_id,
            mix_index,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn blake2b_memory(
        builder: &mut BytesBuilder<L>,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        t_values: &ArrayRegister<Self::IntRegister>,
        const_nums: &BLAKE2BConstNums,
        consts: &BLAKE2BConsts<BytesBuilder<L>>,
        num_messages_element: &ElementRegister,
        num_real_compresses: usize,
        num_real_compresses_element: &ElementRegister,
        num_dummy_rows: usize,
    ) -> BLAKE2BMemory {
        // Initialize the h memory
        let h = builder.uninit_slice();

        // Set dummy reads for h
        // Every row in the first compress of each message will read it 10 times. (96 * 10 * num_messages))
        // For the non first compress of each message
        //    1) First four rows will read it 8 times.  4 * 8
        //    2) Last row will read it 2 times.    2
        //    3) All other rows will read it 10 times.   91 * 10
        // For the dummy compress, it will read it 10 times per row. (length_last_round * 10)
        let num_dummy_rows_element =
            builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(num_dummy_rows));
        let num_non_first_compresses: ElementRegister = builder
            .public_expression(num_real_compresses_element.expr() - num_messages_element.expr());
        let num_dummy_h_reads = builder.public_expression(
            (num_messages_element.expr() * const_nums.const_96.expr() * const_nums.const_10.expr())
                + (num_non_first_compresses.expr()
                    * (const_nums.const_4.expr() * const_nums.const_8.expr()
                        + const_nums.const_2.expr()
                        + const_nums.const_91.expr() * const_nums.const_10.expr()))
                + (num_dummy_rows_element.expr() * const_nums.const_10.expr()),
        );
        builder.store(
            &h.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            Some(num_dummy_h_reads),
            Some("h".to_string()),
            Some(MemorySliceIndex::IndexElement(consts.dummy_index)),
        );

        // Initialize the v memory
        let v = builder.uninit_slice();
        // Set dummy reads for v
        // Every first four rows of every real compress round will read it four times.
        // Every dummy row will read it four times.
        let num_dummy_v_reads = builder.constant::<ElementRegister>(
            &L::Field::from_canonical_usize(num_real_compresses * 16 + num_dummy_rows * 4),
        );
        builder.store(
            &v.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            Some(num_dummy_v_reads),
            Some("v".to_string()),
            Some(MemorySliceIndex::IndexElement(consts.dummy_index)),
        );

        // Initialize the v_final memory
        let v_final = builder.uninit_slice();
        // Set dummy reads for v_final
        // Every first 95 rows of every real compress round will read it 16 times.
        // Every dummy row will read it 16 times.
        let num_dummy_v_final_reads = builder.constant::<ElementRegister>(
            &L::Field::from_canonical_usize(num_real_compresses * 95 * 16 + num_dummy_rows * 16),
        );
        builder.store(
            &v_final.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            Some(num_dummy_v_final_reads),
            Some("v_final".to_string()),
            Some(MemorySliceIndex::IndexElement(consts.dummy_index)),
        );

        // Initialize the m memory
        let m = builder.uninit_slice();

        // Each message chunk will be read 24 times per compress.  Two times per compress iteration.
        for (compress_id_value, padded_chunk) in padded_chunks.iter().enumerate() {
            assert!(padded_chunk.len() == MSG_ARRAY_SIZE);
            for (j, word) in padded_chunk.iter().enumerate() {
                builder.store(
                    &m.get(compress_id_value * 16 + j),
                    word,
                    &Time::zero(),
                    Some(const_nums.const_12),
                    Some("m".to_string()),
                    Some(MemorySliceIndex::Index(compress_id_value * 16 + j)),
                );
            }
        }
        // Set dummy reads for m
        // For each dummy row, it will read it 2 times.
        let num_dummy_m_reads = builder
            .constant::<ElementRegister>(&L::Field::from_canonical_usize(num_dummy_rows * 2));
        builder.store(
            &m.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_m_reads),
            Some("m".to_string()),
            Some(MemorySliceIndex::IndexElement(consts.dummy_index)),
        );

        // Initialize the t memory
        let t = builder.uninit_slice();
        for (compress_id, t_value) in t_values.iter().enumerate() {
            builder.store(
                &t.get(compress_id),
                t_value,
                &Time::zero(),
                Some(const_nums.const_96),
                Some("t".to_string()),
                Some(MemorySliceIndex::Index(compress_id)),
            );
        }
        // Set dummy reads for t.
        // For each dummy row, it will read it once.
        builder.store(
            &t.get_at(consts.dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_rows_element),
            Some("t".to_string()),
            Some(MemorySliceIndex::IndexElement(consts.dummy_index)),
        );

        BLAKE2BMemory {
            h,
            v,
            v_final,
            m,
            t,
        }
    }

    fn blake2b_data(
        builder: &mut BytesBuilder<L>,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        t_values: &ArrayRegister<Self::IntRegister>,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: &ArrayRegister<ElementRegister>,
        num_messages_element: &ElementRegister,
    ) -> BLAKE2BData<BytesBuilder<L>> {
        assert_eq!(padded_chunks.len(), end_bits.len());

        let num_real_compresses = padded_chunks.len();
        debug!("num_real_compresses: {}", num_real_compresses);
        let num_real_compresses_element = builder
            .constant::<ElementRegister>(&L::Field::from_canonical_usize(num_real_compresses));
        let degree_log = log2_ceil(num_real_compresses * 96);
        assert!(degree_log < 31, "AIR degree is too large");
        debug!("AIR degree after padding: {}", 1 << degree_log);

        let num_dummy_compresses = (1 << degree_log) / COMPRESS_LENGTH + 1 - num_real_compresses;
        let length_last_compress = (1 << degree_log) % COMPRESS_LENGTH;
        let length_last_compress_element = builder
            .constant::<ElementRegister>(&L::Field::from_canonical_usize(length_last_compress));
        let num_dummy_rows = (num_dummy_compresses - 1) * COMPRESS_LENGTH + length_last_compress;

        let num_rows_element =
            builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(1 << degree_log));

        // create the const numbers data
        let const_nums = Self::blake2b_const_nums(builder);

        let mut num_total_mixes = (num_real_compresses + num_dummy_compresses - 1) * NUM_MIX_ROUNDS;
        let num_mixes_last_compress = length_last_compress / MIX_LENGTH;
        assert!(num_mixes_last_compress == 4 || num_mixes_last_compress == 8);
        num_total_mixes += num_mixes_last_compress;

        let public = BLAKE2BPublicData {
            padded_chunks: padded_chunks.to_vec(),
            t_values: *t_values,
            end_bits: *end_bits,
            digest_indices: *digest_indices,
        };

        // create the consts data
        let consts = Self::blake2b_const(
            builder,
            &num_rows_element,
            num_messages_element,
            num_real_compresses,
            &num_real_compresses_element,
            num_dummy_compresses,
            num_total_mixes,
            num_mixes_last_compress,
            &const_nums,
        );

        // create the trace data
        let trace = Self::blake2b_trace_data(
            builder,
            &const_nums,
            &consts,
            num_real_compresses,
            end_bits,
            digest_bits,
            num_dummy_compresses,
            length_last_compress,
            &length_last_compress_element,
        );

        // create the memory data
        let memory = Self::blake2b_memory(
            builder,
            padded_chunks,
            t_values,
            &const_nums,
            &consts,
            num_messages_element,
            num_real_compresses,
            &num_real_compresses_element,
            num_dummy_rows,
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
    fn blake2b_compress_initialize(
        builder: &mut BytesBuilder<L>,
        data: &BLAKE2BData<BytesBuilder<L>>,
    ) -> ([ElementRegister; 4], [Self::IntRegister; 4]) {
        let init_idx_1 = data.trace.compress_index;
        let init_idx_2 = builder.add(data.trace.compress_index, data.const_nums.const_4);

        // Read the h values.
        //
        // Read the dummy index from h at any of the following conditions
        // 1) in the first compress of a message (first 4 rows will read from IV instead)
        // 2) NOT in the first 4 rows of compress (e.g. not is_compress_initialize)
        // 3) in the dummy compress.
        //
        // Boolean expression is at_first_compress OR NOT(is_compress_initialize) OR at_dummy_compress
        // That is equivalent to
        // NOT(NOT(at_first_compress) AND is_compress_initialize AND NOT(at_dummy_compress))
        let read_dummy_h_idx = builder.expression(
            data.const_nums.const_1.expr()
                - (data.trace.at_first_compress.not_expr()
                    * data.trace.is_compress_initialize.expr()
                    * data.trace.at_dummy_compress.not_expr()),
        );

        let mut h_idx_1 = builder.expression(
            data.trace.previous_compress_id.expr() * data.const_nums.const_8.expr()
                + init_idx_1.expr(),
        );
        h_idx_1 = builder.select(read_dummy_h_idx, &data.consts.dummy_index, &h_idx_1);

        let mut h_idx_2 = builder.expression(
            data.trace.previous_compress_id.expr() * data.const_nums.const_8.expr()
                + init_idx_2.expr(),
        );
        h_idx_2 = builder.select(read_dummy_h_idx, &data.consts.dummy_index, &h_idx_2);

        let h_ts = builder.select(
            read_dummy_h_idx,
            &data.consts.dummy_ts,
            &data.const_nums.const_0,
        );

        let mut h_value_1 = builder.load(
            &data.memory.h.get_at(h_idx_1),
            &Time::from_element(h_ts),
            Some("h".to_string()),
            Some(MemorySliceIndex::IndexElement(h_idx_1)),
        );
        let mut h_value_2 = builder.load(
            &data.memory.h.get_at(h_idx_2),
            &Time::from_element(h_ts),
            Some("h".to_string()),
            Some(MemorySliceIndex::IndexElement(h_idx_2)),
        );

        // Read the iv values.
        //
        // Read the dummy value at any of the following conditions
        // 1) NOT in the first 4 rows of compress (e.g. not is_compress_initialize)
        // 2) NOT the first compress of a messge
        // 3) In the dummy compress
        //
        // Boolean expression is NOT(is_compress_initialize) OR NOT(at_first_compress) OR at_dummy_compress
        // That is equivalent to
        // NOT(is_compress_initialize AND at_first_compress AND NOT(at_dummy_compress))
        let read_dummy_iv_idx = builder.expression(
            data.const_nums.const_1.expr()
                - (data.trace.is_compress_initialize.expr()
                    * data.trace.at_first_compress.expr()
                    * data.trace.at_dummy_compress.not_expr()),
        );
        let iv_idx_1 = builder.select(read_dummy_iv_idx, &data.consts.dummy_index, &init_idx_1);
        let iv_idx_2 = builder.select(read_dummy_iv_idx, &data.consts.dummy_index, &init_idx_2);

        let iv_value_1 = builder.load(
            &data.consts.iv.get_at(iv_idx_1),
            &Time::zero(),
            Some("iv".to_string()),
            Some(MemorySliceIndex::IndexElement(iv_idx_1)),
        );
        let iv_value_2 = builder.load(
            &data.consts.iv.get_at(iv_idx_2),
            &Time::zero(),
            Some("iv".to_string()),
            Some(MemorySliceIndex::IndexElement(iv_idx_2)),
        );

        // Read the compress iv values.
        //
        // Read the dummy value at any of the following conditions
        // 1) NOT in the first 4 rows of compress (e.g. not is_compress_initialize)
        // 2) In the dummy compress
        //
        // Boolean expression is NOT(is_compress_initialize) OR at_dummy_compress
        // That is equivalent to
        // NOT(is_compress_initialize AND NOT(at_dummy_compress))

        let read_dummy_compress_iv_idx = builder.expression(
            data.const_nums.const_1.expr()
                - (data.trace.is_compress_initialize.expr()
                    * data.trace.at_dummy_compress.not_expr()),
        );
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
            Some("compress_iv".to_string()),
            Some(MemorySliceIndex::IndexElement(compress_iv_idx_1)),
        );
        let compress_iv_value_2 = builder.load(
            &data.consts.compress_iv.get_at(compress_iv_idx_2),
            &Time::zero(),
            Some("compress_iv".to_string()),
            Some(MemorySliceIndex::IndexElement(compress_iv_idx_2)),
        );

        // Read the v values.
        //
        // First get the v indicies and last write timestamps.
        let v_indices = &data.consts.v_indices;
        let v1_idx = v_indices.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_0,
            Some("mix_index".to_string()),
        );
        let v2_idx = v_indices.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_1,
            Some("mix_index".to_string()),
        );
        let v3_idx = v_indices.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_2,
            Some("mix_index".to_string()),
        );
        let v4_idx = v_indices.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_3,
            Some("mix_index".to_string()),
        );

        let v_last_write_ages = &data.consts.v_last_write_ages;
        let v1_last_write_age = v_last_write_ages.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_0,
            Some("v_last_write_ages".to_string()),
        );
        let v2_last_write_age = v_last_write_ages.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_1,
            Some("v_last_write_ages".to_string()),
        );
        let v3_last_write_age = v_last_write_ages.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_2,
            Some("v_last_write_ages".to_string()),
        );
        let v4_last_write_age = v_last_write_ages.get_at(
            builder,
            data.trace.mix_index,
            data.const_nums.const_3,
            Some("v_last_write_ages".to_string()),
        );

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
        // 2) In the dummy compress
        let read_dummy_v_idx = builder.or(
            data.trace.is_compress_initialize,
            data.trace.at_dummy_compress,
        );

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
            Some("v".to_string()),
            Some(MemorySliceIndex::IndexElement(v1_read_idx)),
        );
        let mut v2_value = builder.load(
            &data.memory.v.get_at(v2_read_idx),
            &Time::from_element(v2_last_write_ts),
            Some("v".to_string()),
            Some(MemorySliceIndex::IndexElement(v2_read_idx)),
        );
        let mut v3_value = builder.load(
            &data.memory.v.get_at(v3_read_idx),
            &Time::from_element(v3_last_write_ts),
            Some("v".to_string()),
            Some(MemorySliceIndex::IndexElement(v3_read_idx)),
        );
        let mut v4_value = builder.load(
            &data.memory.v.get_at(v4_read_idx),
            &Time::from_element(v4_last_write_ts),
            Some("v".to_string()),
            Some(MemorySliceIndex::IndexElement(v4_read_idx)),
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
        let t_idx = builder.select(
            data.trace.at_dummy_compress,
            &data.consts.dummy_index,
            &data.trace.compress_id,
        );
        let t = builder.load(
            &data.memory.t.get_at(t_idx),
            &Time::zero(),
            Some("t".to_string()),
            Some(MemorySliceIndex::IndexElement(t_idx)),
        );

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
    fn blake2b_compress(
        builder: &mut BytesBuilder<L>,
        v_indices: &[ElementRegister; 4],
        v_values: &[Self::IntRegister; 4],
        data: &BLAKE2BData<BytesBuilder<L>>,
    ) {
        // Load the permutation values.
        let mut permutation_col: ElementRegister =
            builder.mul(data.trace.mix_index, data.const_nums.const_2);

        let mut m_idx_1 = data.consts.permutations.get_at(
            builder,
            data.trace.mix_id,
            permutation_col,
            Some("permutation".to_string()),
        );

        m_idx_1 = builder.expression(
            data.trace.compress_id.expr() * data.const_nums.const_16.expr() + m_idx_1.expr(),
        );
        permutation_col = builder.add(permutation_col, data.const_nums.const_1);

        let mut m_idx_2 = data.consts.permutations.get_at(
            builder,
            data.trace.mix_id,
            permutation_col,
            Some("permutation".to_string()),
        );

        m_idx_2 = builder.expression(
            data.trace.compress_id.expr() * data.const_nums.const_16.expr() + m_idx_2.expr(),
        );

        m_idx_1 = builder.select(
            data.trace.at_dummy_compress,
            &data.consts.dummy_index,
            &m_idx_1,
        );
        m_idx_2 = builder.select(
            data.trace.at_dummy_compress,
            &data.consts.dummy_index,
            &m_idx_2,
        );

        // Load the message values.
        let m_1 = builder.load(
            &data.memory.m.get_at(m_idx_1),
            &Time::zero(),
            Some("m".to_string()),
            Some(MemorySliceIndex::IndexElement(m_idx_1)),
        );
        let m_2 = builder.load(
            &data.memory.m.get_at(m_idx_2),
            &Time::zero(),
            Some("m".to_string()),
            Some(MemorySliceIndex::IndexElement(m_idx_2)),
        );

        // Output the "parameters" being sent to the mix function.

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
        // 2) NOT in the dummy compress.
        //
        // Boolean expression is NOT(is_compress_initialize) AND NOT(at_dummy_compress)
        let save_v = builder.expression(
            data.trace.is_compress_finalize.not_expr() * data.trace.at_dummy_compress.not_expr(),
        );

        // Save the output into v in all of the following conditions.
        // 1) in the last 4 rows of compress (e.g. is_compress_finalize)
        // 2) NOT in the dummy compress.
        //
        // Boolean expression is is_compress_finalize AND NOT(at_dummy_compress)
        let save_v_final = builder.expression(
            data.trace.is_compress_finalize.expr() * data.trace.at_dummy_compress.not_expr(),
        );

        let updated_v_values = [updated_v0, updated_v1, updated_v2, updated_v3];
        let clk = builder.clk;
        for (value, v_index) in updated_v_values.iter().zip(v_indices.iter()) {
            let v_idx = builder.select(save_v, v_index, &data.consts.dummy_index_2);
            let v_value = builder.select(save_v, value, &data.const_nums.const_0_u64);
            let v_ts = builder.select(save_v, &clk, &data.consts.dummy_ts);

            builder.store(
                &data.memory.v.get_at(v_idx),
                v_value,
                &Time::from_element(v_ts),
                Some(save_v.as_element()),
                Some("v".to_string()),
                Some(MemorySliceIndex::IndexElement(v_idx)),
            );

            let v_final_idx = builder.select(save_v_final, v_index, &data.consts.dummy_index_2);
            let v_final_ts =
                builder.select(save_v_final, &data.trace.compress_id, &data.consts.dummy_ts);
            let v_final_value = builder.select(save_v_final, value, &data.const_nums.const_0_u64);

            builder.store(
                &data.memory.v_final.get_at(v_final_idx),
                v_final_value,
                &Time::from_element(v_final_ts),
                Some(save_v_final.as_element()),
                Some("v_final".to_string()),
                Some(MemorySliceIndex::IndexElement(v_final_idx)),
            );
        }
    }

    fn blake2b_compress_finalize(
        builder: &mut BytesBuilder<L>,
        state_ptr: &Slice<Self::IntRegister>,
        data: &BLAKE2BData<BytesBuilder<L>>,
    ) {
        // If we are at the last row of compress, then compute and save the h value.

        // First load the previous round's h value.
        let h_workspace_1 = builder.alloc_array::<Self::IntRegister>(STATE_SIZE);

        // Read dummy h values if any of the following conditions are true
        // 1) NOT at last row of a compress
        // 2) at the first compress
        // 3) at a dummy compress
        //
        // Boolean expression is NOT(is_compress_final_row) OR at_first_compress OR at_dummy_compress
        // That is equivalent to
        // NOT(is_compress_final_row AND NOT(at_first_compress) AND NOT(at_dummy_compress))
        let read_dummy_h_idx = builder.expression(
            data.const_nums.const_1.expr()
                - (data.trace.is_compress_final_row.expr()
                    * data.trace.at_first_compress.not_expr()
                    * data.trace.at_dummy_compress.not_expr()),
        );

        let h_ts = builder.select(
            read_dummy_h_idx,
            &data.consts.dummy_ts,
            &data.const_nums.const_0,
        );
        for i in 0..STATE_SIZE {
            let i_element = builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(i));
            let mut h_idx = builder.expression(
                data.trace.previous_compress_id.expr() * data.const_nums.const_8.expr()
                    + i_element.expr(),
            );
            h_idx = builder.select(read_dummy_h_idx, &data.consts.dummy_index, &h_idx);
            let mut h_value = builder.load(
                &data.memory.h.get_at(h_idx),
                &Time::from_element(h_ts),
                Some("h".to_string()),
                Some(MemorySliceIndex::IndexElement(h_idx)),
            );

            // If we are at the first compress of a message, then use the iv values instead of the h values.
            h_value = builder.select(
                data.trace.at_first_compress,
                &data.consts.iv_values.get(i),
                &h_value,
            );
            builder.set_to_expression(&h_workspace_1.get(i), h_value.expr());
        }

        // Xor the first 8 final v values
        let h_workspace_2 = builder.alloc_array::<Self::IntRegister>(STATE_SIZE);

        // Read dummy v_final values if NOT at last row of a compress OR in a dummy compress.
        //
        // Boolean expression is NOT(is_compress_final_row) OR at_dummy_compress
        // That is equivalent to
        // NOT(is_compress_final_row AND NOT(at_dummy_compress))
        let read_dummy_v_final_idx = builder.expression(
            data.const_nums.const_1.expr()
                - (data.trace.is_compress_final_row.expr()
                    * data.trace.at_dummy_compress.not_expr()),
        );
        let v_final_ts = builder.select(
            read_dummy_v_final_idx,
            &data.consts.dummy_ts,
            &data.trace.compress_id,
        );
        for i in 0..STATE_SIZE {
            let i_element = builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(i));
            let v_final_idx =
                builder.select(read_dummy_v_final_idx, &data.consts.dummy_index, &i_element);
            let v_i = builder.load(
                &data.memory.v_final.get_at(v_final_idx),
                &Time::from_element(v_final_ts),
                Some("v_final".to_string()),
                Some(MemorySliceIndex::IndexElement(v_final_idx)),
            );
            let updated_h = builder.xor(h_workspace_1.get(i), v_i);
            builder.set_to_expression(&h_workspace_2.get(i), updated_h.expr());
        }

        // Xor the second 8 final v values
        let h = builder.alloc_array::<Self::IntRegister>(STATE_SIZE);

        // Save h into memory if we are at the final row and it is not the end compress and not in a dummy compress.
        let save_h = builder.expression(
            data.trace.is_compress_final_row.expr()
                * data.trace.at_end_compress.not_expr()
                * data.trace.at_dummy_compress.not_expr(),
        );
        for i in 0..STATE_SIZE {
            let i_element = builder.constant::<ElementRegister>(&L::Field::from_canonical_usize(i));
            let i_element_plus_8 = builder.add(i_element, data.const_nums.const_8);

            let v_final_idx = builder.select(
                read_dummy_v_final_idx,
                &data.consts.dummy_index,
                &i_element_plus_8,
            );

            let v_value = builder.load(
                &data.memory.v_final.get_at(v_final_idx),
                &Time::from_element(v_final_ts),
                Some("v_final".to_string()),
                Some(MemorySliceIndex::IndexElement(v_final_idx)),
            );
            let xor = builder.xor(h_workspace_2.get(i), v_value);
            builder.set_to_expression(&h.get(i), xor.expr());

            let mut h_idx = builder.expression(
                data.trace.compress_id.expr() * data.const_nums.const_8.expr() + i_element.expr(),
            );
            h_idx = builder.select(save_h, &h_idx, &data.consts.dummy_index_2);
            let h_value = builder.select(save_h, &xor, &data.const_nums.const_0_u64);
            let h_ts = builder.select(save_h, &data.const_nums.const_0, &data.consts.dummy_ts);
            // Need to save the h value twice as it will be written twice per compress.
            let h_multiplicity =
                builder.select(save_h, &data.const_nums.const_2, &data.const_nums.const_0);

            builder.store(
                &data.memory.h.get_at(h_idx),
                h_value,
                &Time::from_element(h_ts),
                Some(h_multiplicity),
                Some("h".to_string()),
                Some(MemorySliceIndex::IndexElement(h_idx)),
            );

            // If this is the digest row, then also store the calculated digest.
            // Only need to do so for the first 4 entries of h.
            if i < 4 {
                builder.store(
                    &state_ptr.get(i),
                    xor,
                    &Time::from_element(data.trace.compress_id),
                    Some(data.trace.is_digest_row.as_element()),
                    Some("state_ptr".to_string()),
                    Some(MemorySliceIndex::Index(i)),
                );
            }
        }
    }

    fn blake2b_mix(
        builder: &mut BytesBuilder<L>,
        v_a: &Self::IntRegister,
        v_b: &Self::IntRegister,
        v_c: &Self::IntRegister,
        v_d: &Self::IntRegister,
        x: &Self::IntRegister,
        y: &Self::IntRegister,
    ) -> (
        Self::IntRegister,
        Self::IntRegister,
        Self::IntRegister,
        Self::IntRegister,
    ) {
        let mut v_a_inter = builder.add(*v_a, *v_b);
        v_a_inter = builder.add(v_a_inter, *x);

        let mut v_d_inter = builder.xor(*v_d, v_a_inter);
        v_d_inter = builder.rotate_right(v_d_inter, 32);

        let mut v_c_inter = builder.add(*v_c, v_d_inter);

        let mut v_b_inter = builder.xor(*v_b, v_c_inter);
        v_b_inter = builder.rotate_right(v_b_inter, 24);

        v_a_inter = builder.add(v_a_inter, v_b_inter);
        v_a_inter = builder.add(v_a_inter, *y);

        v_d_inter = builder.xor(v_d_inter, v_a_inter);
        v_d_inter = builder.rotate_right(v_d_inter, 16);

        v_c_inter = builder.add(v_c_inter, v_d_inter);

        v_b_inter = builder.xor(v_b_inter, v_c_inter);
        v_b_inter = builder.rotate_right(v_b_inter, 63);

        (v_a_inter, v_b_inter, v_c_inter, v_d_inter)
    }
}
