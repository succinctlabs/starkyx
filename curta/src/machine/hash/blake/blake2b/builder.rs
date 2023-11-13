use super::data::{BLAKE2BConstNums, BLAKE2BConsts, BLAKE2BData};
use super::IV;
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
    COMPRESS_INITIALIZE_INDICES, COMPRESS_IV, SIGMA_PERMUTATIONS, V_INDICES, V_LAST_WRITE_AGES,
};
use crate::math::prelude::*;

const DUMMY_INDEX: u64 = i32::MAX as u64;
const DUMMY_TS: u64 = i32::MAX as u64;

impl<L: AirParameters> BytesBuilder<L>
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
        &mut self,
        padded_chunks: &[ArrayRegister<U64Register>],
        msg_lens: &[ArrayRegister<U64Register>],
        end_bits: &ArrayRegister<BitRegister>,
    ) {
        let data = self.blake2b_data(padded_chunks, msg_lens, end_bits);

        let (v_indices, v_values) = self.blake2b_compress_initialize(&data);
        self.blake2b_compress(&v_indices, &v_values, &data);
    }

    pub fn blake2b_const_nums(&mut self) -> BLAKE2BConstNums {
        BLAKE2BConstNums {
            const_0: self.constant(&L::Field::from_canonical_u8(0)),
            const_0_u64: self.constant(&u64_to_le_field_bytes(0u64)),
            const_1: self.constant(&L::Field::from_canonical_u8(1)),
            const_2: self.constant(&L::Field::from_canonical_u8(2)),
            const_3: self.constant(&L::Field::from_canonical_u8(3)),
            const_4: self.constant(&L::Field::from_canonical_u8(4)),
            const_5: self.constant(&L::Field::from_canonical_u8(5)),
            const_6: self.constant(&L::Field::from_canonical_u8(6)),
            const_7: self.constant(&L::Field::from_canonical_u8(7)),
            const_8: self.constant(&L::Field::from_canonical_u8(8)),
            const_9: self.constant(&L::Field::from_canonical_u8(9)),
            const_10: self.constant(&L::Field::from_canonical_u8(10)),
            const_11: self.constant(&L::Field::from_canonical_u8(11)),
            const_12: self.constant(&L::Field::from_canonical_u8(12)),
            const_13: self.constant(&L::Field::from_canonical_u8(13)),
            const_14: self.constant(&L::Field::from_canonical_u8(14)),
            const_15: self.constant(&L::Field::from_canonical_u8(15)),
            const_96: self.constant(&L::Field::from_canonical_u8(96)),
            const_ffffffffffffffff: self
                .constant::<U64Register>(&u64_to_le_field_bytes::<L::Field>(0xFFFFFFFFFFFFFFFF)),
        }
    }

    pub fn blake2b_const(
        &mut self,
        num_compress_element: &ElementRegister,
        num_mix_iterations: &ElementRegister,
        end_bits: &ArrayRegister<BitRegister>,
        const_nums: &BLAKE2BConstNums,
    ) -> BLAKE2BConsts<L> {
        let mut num_messages: ElementRegister = self.alloc_public();
        for end_bit in end_bits.iter() {
            num_messages = self.expression(end_bit.expr() + num_messages.expr());
        }

        assert!(DUMMY_INDEX < L::Field::order());
        let dummy_index: ElementRegister =
            self.constant(&L::Field::from_canonical_u64(DUMMY_INDEX));

        assert!(DUMMY_TS < L::Field::order());
        let dummy_ts: ElementRegister = self.constant(&L::Field::from_canonical_u64(DUMMY_TS));

        let iv_values = self.constant_array::<U64Register>(&IV.map(u64_to_le_field_bytes));
        let iv = self.uninit_slice();
        for (i, value) in iv_values.iter().enumerate() {
            self.store(&iv.get(i), value, &Time::zero(), Some(num_messages));
        }
        let num_dummy_reads = 2 * (96 - 4) * 2;
        let num_dummy_reads_element =
            self.constant(&L::Field::from_canonical_usize(num_dummy_reads));
        self.store(
            &iv.get_at(dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_reads_element),
        );

        let compress_iv_values =
            self.constant_array::<U64Register>(&COMPRESS_IV.map(u64_to_le_field_bytes));
        let compress_iv = self.uninit_slice();
        for (i, value) in compress_iv_values.iter().enumerate() {
            self.store(
                &compress_iv.get(i),
                value,
                &Time::zero(),
                Some(*num_compress_element),
            );
        }
        let num_dummy_reads = 2 * (96 - 4) * 2;
        let num_dummy_reads_element =
            self.constant(&L::Field::from_canonical_usize(num_dummy_reads));
        self.store(
            &compress_iv.get_at(dummy_index),
            const_nums.const_0_u64,
            &Time::zero(),
            Some(num_dummy_reads_element),
        );

        let mut compress_initial_indices = MemoryArray::<L, 4, 2>::new(self);
        for (i, indices) in COMPRESS_INITIALIZE_INDICES.iter().enumerate() {
            compress_initial_indices.store_row(self, i, indices, *num_compress_element);
        }
        // Add in the stores for the dummy index.  There will be two reads for all compress rows
        // other than the first four.
        let num_dummy_reads = 2 * (96 - 4);
        let num_dummy_reads_element =
            self.constant(&L::Field::from_canonical_usize(num_dummy_reads));
        self.store(
            &compress_initial_indices
                .flattened_memory
                .get_at(dummy_index),
            dummy_index,
            &Time::zero(),
            Some(num_dummy_reads_element),
        );

        // Each element is loaded once per compress cycle.
        let mut v_indices = MemoryArray::<L, 8, 4>::new(self);
        for (i, indices) in V_INDICES.iter().enumerate() {
            v_indices.store_row(self, i, indices, *num_mix_iterations);
        }

        let mut v_last_write_ages = MemoryArray::<L, 8, 4>::new(self);
        for (i, ages) in V_LAST_WRITE_AGES.iter().enumerate() {
            v_last_write_ages.store_row(self, i, ages, *num_mix_iterations);
        }

        let mut permutations = MemoryArray::<L, 12, 16>::new(self);
        for (i, permutation) in SIGMA_PERMUTATIONS.iter().enumerate() {
            permutations.store_row(self, i, permutation, *num_mix_iterations);
        }

        BLAKE2BConsts {
            iv,
            compress_iv,
            compress_initial_indices,
            v_indices,
            v_last_write_ages,
            permutations,
            dummy_index,
            dummy_ts,
        }
    }

    // This function will create all the registers/memory slots that will be used for control flow
    // related functions.
    pub fn blake2b_trace_data(
        &mut self,
        const_nums: &BLAKE2BConstNums,
        num_rounds_element: &ElementRegister,
        end_bits: &ArrayRegister<BitRegister>,
    ) -> BLAKE2BTraceData {
        let (cycle_3_end_bit, cycle_4_end_bit, cycle_8_end_bit, cycle_96_end_bit) =
            Self::cycles_end_bits(self);

        // Allocate end_bits from public input.
        let end_bit = self.uninit_slice();
        for (i, end_bit_val) in end_bits.iter().enumerate() {
            self.store(
                &end_bit.get(i),
                end_bit_val,
                &Time::zero(),
                Some(const_nums.const_96),
            );
        }

        // `compress_id` is a register is computed by counting the number of cycles. We do this by
        // setting `process_id` to be the cumulative sum of the `end_bit` of each cycle.
        let compress_id: ElementRegister = self.alloc::<ElementRegister>();
        self.set_to_expression_first_row(&compress_id, L::Field::ZERO.into());
        self.set_to_expression_transition(
            &compress_id.next(),
            compress_id.expr() + cycle_96_end_bit.expr(),
        );

        let mix_index = self.alloc::<ElementRegister>();
        self.set_to_expression_first_row(&mix_index, L::Field::ZERO.into());
        self.set_to_expression_transition(
            &mix_index.next(),
            cycle_8_end_bit.not_expr() * (mix_index.expr() + const_nums.const_1.expr())
                + cycle_8_end_bit.expr() * const_nums.const_9.expr(),
        );

        // The array index register can be computed as `clock - process_id * CYCLE_LENGTH`.
        let clk = self.clk;
        let compress_index =
            self.expression(clk.expr() - compress_id.expr() * const_nums.const_96.expr());

        // Flag if we are within the first four rows of a hash invocation.  In these rows, we will
        // need to use the IV values.
        let is_hash_initialize = self.alloc::<BitRegister>();
        self.set_to_expression_first_row(&is_hash_initialize, L::Field::ONE.into());
        let at_last_hash_compress = self.load(&end_bit.get_at(compress_id), &Time::zero());
        self.set_to_expression_transition(
            &is_hash_initialize.next(),
            (cycle_96_end_bit.expr() * at_last_hash_compress.expr())
                + (cycle_4_end_bit.not_expr() * is_hash_initialize.expr()
                    + cycle_4_end_bit.expr() * const_nums.const_0.expr()),
        );

        // Flag if we are within the first four rows of a compress.  In these rows, we will need to
        // use the COMPRESS_IV values.
        let is_compress_initialize = self.alloc::<BitRegister>();
        self.set_to_expression_first_row(&is_compress_initialize, L::Field::ONE.into());
        self.set_to_expression_transition(
            &is_compress_initialize.next(),
            (cycle_96_end_bit.expr() * const_nums.const_1.expr())
                + (cycle_96_end_bit.not_expr()
                    * (cycle_4_end_bit.expr() * const_nums.const_0.expr()
                        + cycle_4_end_bit.not_expr() * is_compress_initialize.expr())),
        );

        // Flag if we are in the first row of a hash.  In that case, we will need to do an
        // xor for the v_12 value.
        let is_compress_first_row = self.alloc::<BitRegister>();
        self.set_to_expression_first_row(&is_compress_first_row, L::Field::ONE.into());
        self.set_to_expression_transition(&is_compress_first_row.next(), cycle_96_end_bit.expr());

        // Flag if we are in the 3rd row of a hash.  In that case, we will need to do a xor on
        // the v_14 value.
        let is_compress_third_row =
            self.expression(is_compress_initialize.expr() * cycle_3_end_bit.expr());

        // Need to flag to the last 4 rows of the compress cycle.
        // At those rows, the V values should be saved to v_final, so that those values can be used
        // to calculate the compress h values.
        let save_h: Slice<BitRegister> = self.uninit_slice();
        let true_const = self.constant::<BitRegister>(&L::Field::from_canonical_usize(1));
        let false_const = self.constant::<BitRegister>(&L::Field::from_canonical_usize(0));
        for i in 0..96 {
            self.store(
                &save_h.get(i),
                if i < 92 { false_const } else { true_const },
                &Time::zero(),
                Some(*num_rounds_element),
            );
        }

        BLAKE2BTraceData {
            clk,
            is_hash_initialize,
            is_compress_initialize,
            is_compress_first_row,
            is_compress_third_row,
            save_h,
            compress_id,
            compress_index,
            mix_index,
        }
    }

    pub fn blake2b_memory(
        &mut self,
        padded_chunks: &[ArrayRegister<U64Register>],
        num_consts: &BLAKE2BConstNums,
        consts: &BLAKE2BConsts<L>,
    ) -> BLAKE2BMemory {
        // Initialize the h memory
        // Need to set DUMMY_VALUE at DUMMY_TS with multiplicity of (96 - 4) * 2.
        let num_dummy_accesses = self.constant(&L::Field::from_canonical_usize(368));
        let h = self.uninit_slice();
        self.store(
            &h.get_at(consts.dummy_index),
            num_consts.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            Some(num_dummy_accesses),
        );

        // Initialize the v memory
        // Need to set DUMMY_VALUE at DUMMY_TS with multiplicity of 1.
        let v = self.uninit_slice();
        self.store(
            &v.get_at(consts.dummy_index),
            num_consts.const_0_u64,
            &Time::from_element(consts.dummy_ts),
            None,
        );

        // Initialize the v final memory
        let v_final = self.uninit_slice();

        // Initialize the m memory
        let m = self.uninit_slice();

        for (i, padded_chunk) in padded_chunks.iter().enumerate() {
            for (j, word) in padded_chunk.iter().enumerate().take(32) {
                self.store(&m.get(j), word, &Time::constant(i), None);
            }
        }

        BLAKE2BMemory { h, v, v_final, m }
    }

    pub fn blake2b_data(
        &mut self,
        padded_chunks: &[ArrayRegister<U64Register>],
        msg_lens: &[ArrayRegister<U64Register>],
        end_bits: &ArrayRegister<BitRegister>,
    ) -> BLAKE2BData<L> {
        assert_eq!(padded_chunks.len(), end_bits.len());

        // create the const numbers data
        let const_nums = self.blake2b_const_nums();

        let num_compresses = padded_chunks.len();
        // Convert the number of rounds to a field element.
        let num_compresses_element = self.constant(&L::Field::from_canonical_usize(num_compresses));

        let num_mix_iterations = num_compresses * 12;
        let num_mix_iterations_element =
            self.constant(&L::Field::from_canonical_usize(num_mix_iterations));

        let public = BLAKE2BPublicData {
            padded_chunks: padded_chunks.to_vec(),
            end_bits: *end_bits,
        };

        // create the consts data
        let consts = self.blake2b_const(
            &num_compresses_element,
            &num_mix_iterations_element,
            end_bits,
            &const_nums,
        );

        // create the trace data
        let trace = self.blake2b_trace_data(&const_nums, &num_compresses_element, end_bits);

        // create the memory data
        let memory = self.blake2b_memory(padded_chunks, &const_nums, &consts);

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
        &mut self,
        data: &BLAKE2BData<L>,
    ) -> ([ElementRegister; 4], [U64Register; 4]) {
        let is_compress_initialize = data.trace.is_compress_initialize;

        // Get the v values if we are within the initialization section of compress (the first four
        // cycles of compress).
        let compress_index = self.select(
            is_compress_initialize,
            &data.trace.compress_index,
            &data.consts.dummy_index,
        );

        let init_idx_1 = &data.consts.compress_initial_indices.get_at(
            self,
            compress_index,
            data.const_nums.const_1,
        );

        let init_idx_2 = &data.consts.compress_initial_indices.get_at(
            self,
            compress_index,
            data.const_nums.const_2,
        );

        let mut previous_compress_id =
            self.expression(data.trace.compress_id.expr() - data.const_nums.const_1.expr());

        // If we are within hash initialize, then read from a dummy h values from a dummy timestamp.
        previous_compress_id = self.select(
            data.trace.is_hash_initialize,
            &data.consts.dummy_ts,
            &previous_compress_id,
        );

        let h_value_1 = self.load(
            &data.memory.h.get_at(*init_idx_1),
            &Time::from_element(previous_compress_id),
        );
        let h_value_2 = self.load(
            &data.memory.h.get_at(*init_idx_2),
            &Time::from_element(previous_compress_id),
        );

        let iv_value_1 = self.load(&data.consts.iv.get_at(*init_idx_1), &Time::zero());
        let iv_value_2 = self.load(&data.consts.iv.get_at(*init_idx_2), &Time::zero());

        let compress_iv_value_1 =
            self.load(&data.consts.compress_iv.get_at(*init_idx_1), &Time::zero());
        let compress_iv_value_2 =
            self.load(&data.consts.compress_iv.get_at(*init_idx_2), &Time::zero());

        // For all the other cycles of compress, read the v values from the v memory. Will need to
        // specify the age of the last write to the v memory entry.
        let v_indices = &data.consts.v_indices;
        let v1_idx = v_indices.get_at(self, data.trace.mix_index, data.const_nums.const_0);
        let v2_idx = v_indices.get_at(self, data.trace.mix_index, data.const_nums.const_1);
        let v3_idx = v_indices.get_at(self, data.trace.mix_index, data.const_nums.const_2);
        let v4_idx = v_indices.get_at(self, data.trace.mix_index, data.const_nums.const_3);

        let v_last_write_ages = &data.consts.v_last_write_ages;
        let v1_last_write_age =
            v_last_write_ages.get_at(self, data.trace.mix_index, data.const_nums.const_0);
        let v2_last_write_age =
            v_last_write_ages.get_at(self, data.trace.mix_index, data.const_nums.const_1);
        let v3_last_write_age =
            v_last_write_ages.get_at(self, data.trace.mix_index, data.const_nums.const_2);
        let v4_last_write_age =
            v_last_write_ages.get_at(self, data.trace.mix_index, data.const_nums.const_3);

        let mut v1_last_write_ts =
            self.expression(data.trace.clk.expr() - v1_last_write_age.expr());
        let mut v2_last_write_ts =
            self.expression(data.trace.clk.expr() - v2_last_write_age.expr());
        let mut v3_last_write_ts =
            self.expression(data.trace.clk.expr() - v3_last_write_age.expr());
        let mut v4_last_write_ts =
            self.expression(data.trace.clk.expr() - v4_last_write_age.expr());

        v1_last_write_ts = self.select(
            data.trace.is_hash_initialize,
            &data.consts.dummy_ts,
            &v1_last_write_ts,
        );

        v2_last_write_ts = self.select(
            data.trace.is_hash_initialize,
            &data.consts.dummy_ts,
            &v2_last_write_ts,
        );

        v3_last_write_ts = self.select(
            data.trace.is_hash_initialize,
            &data.consts.dummy_ts,
            &v3_last_write_ts,
        );

        v4_last_write_ts = self.select(
            data.trace.is_hash_initialize,
            &data.consts.dummy_ts,
            &v4_last_write_ts,
        );

        let v1_value = self.load(
            &data.memory.v.get_at(v1_idx),
            &Time::from_element(v1_last_write_ts),
        );
        let v2_value = self.load(
            &data.memory.v.get_at(v2_idx),
            &Time::from_element(v2_last_write_ts),
        );
        let v3_value = self.load(
            &data.memory.v.get_at(v3_idx),
            &Time::from_element(v3_last_write_ts),
        );
        let v4_value = self.load(
            &data.memory.v.get_at(v4_idx),
            &Time::from_element(v4_last_write_ts),
        );

        let v1_value = self.expression(
            data.trace.is_hash_initialize.expr() * iv_value_1.expr()
                + (data.trace.is_hash_initialize.not_expr()
                    * (data.trace.is_compress_initialize.expr() * h_value_1.expr()
                        + data.trace.is_compress_initialize.not_expr() * v1_value.expr())),
        );

        let v2_value = self.expression(
            data.trace.is_hash_initialize.expr() * iv_value_2.expr()
                + (data.trace.is_hash_initialize.not_expr()
                    * (data.trace.is_compress_initialize.expr() * h_value_2.expr()
                        + data.trace.is_compress_initialize.not_expr() * v2_value.expr())),
        );

        let v3_value = self.expression(
            data.trace.is_compress_initialize.expr() * compress_iv_value_1.expr()
                + data.trace.is_compress_initialize.not_expr() * v3_value.expr(),
        );

        let mut v4_value = self.expression(
            data.trace.is_compress_initialize.expr() * compress_iv_value_2.expr()
                + data.trace.is_compress_initialize.not_expr() * v4_value.expr(),
        );

        // If we are at the first compress row, then will need to xor v4 with t
        // todo!();

        // If we are at the third compress row, then will need to xor v4 with 0xFFFFFFFFFFFFFFFF
        let inverse_v4_value = self.xor(&v4_value, &data.const_nums.const_ffffffffffffffff);
        v4_value = self.select(
            data.trace.is_compress_third_row,
            &inverse_v4_value,
            &v4_value,
        );

        (
            [v1_idx, v2_idx, v3_idx, v4_idx],
            [v1_value, v2_value, v3_value, v4_value],
        )
    }

    /// The processing step of a SHA256 round.
    pub fn blake2b_compress(
        &mut self,
        v_indices: &[ElementRegister; 4],
        v_values: &[U64Register; 4],
        data: &BLAKE2BData<L>,
    ) {
        let m_idx_1 =
            data.consts
                .permutations
                .get_at(self, data.trace.compress_index, data.trace.mix_index);

        let next_col =
            self.expression(data.trace.mix_index.expr() + data.const_nums.const_1.expr());
        let m_idx_2 = data
            .consts
            .permutations
            .get_at(self, data.trace.compress_index, next_col);

        let m_1 = self.load(&data.memory.m.get_at(m_idx_1), &Time::zero());
        let m_2 = self.load(&data.memory.m.get_at(m_idx_2), &Time::zero());

        let (updated_v0, updated_v1, updated_v2, updated_v3) = self.blake2b_mix(
            &v_values[0],
            &v_values[1],
            &v_values[2],
            &v_values[3],
            &m_1,
            &m_2,
        );

        let save_h = self.load(
            &data.trace.save_h.get_at(data.trace.compress_index),
            &Time::zero(),
        );

        let write_ts = self.select(save_h, &data.trace.compress_id, &data.consts.dummy_ts);
        let updated_v_values = [updated_v0, updated_v1, updated_v2, updated_v3];
        for (i, value) in updated_v_values.iter().enumerate() {
            self.store(
                &data.memory.v.get_at(v_indices[i]),
                *value,
                &Time::from_element(self.clk),
                None,
            );

            // Save to v_final if at the last 4 cycles of the round
            let v_final_value = self.select(save_h, value, &data.const_nums.const_ffffffffffffffff);
            self.store(
                &data.memory.v_final.get_at(v_indices[i]),
                v_final_value,
                &Time::from_element(write_ts),
                Some(save_h.as_element()),
            );
        }

        // If we are at the last cycle of the round, then compute and save the h value.

        // First load the previous round's h value.
        let previous_compress_id =
            self.expression(data.trace.compress_id.expr() - data.const_nums.const_1.expr());
        let h_workspace_1 = self.alloc_array::<U64Register>(8);
        let consts = [
            data.const_nums.const_0,
            data.const_nums.const_1,
            data.const_nums.const_2,
            data.const_nums.const_3,
            data.const_nums.const_4,
            data.const_nums.const_5,
            data.const_nums.const_6,
            data.const_nums.const_7,
        ];
        for (i, const_i) in consts.iter().enumerate() {
            let h_value = self.load(
                &data.memory.h.get_at(*const_i),
                &Time::from_element(previous_compress_id),
            );
            self.set_to_expression(&h_workspace_1.get(i), h_value.expr());
        }

        // Xor the first 8 final v values
        let h_workspace_2 = self.alloc_array::<U64Register>(8);
        for (i, const_i) in consts.iter().enumerate() {
            let v_i = self.load(
                &data.memory.v_final.get_at(*const_i),
                &Time::from_element(write_ts),
            );
            let updated_h = self.xor(h_workspace_1.get(i), v_i);
            self.set_to_expression(&h_workspace_2.get(i), updated_h.expr());
        }

        // Xor the second 8 final v values
        let v_indices = [
            data.const_nums.const_8,
            data.const_nums.const_9,
            data.const_nums.const_10,
            data.const_nums.const_11,
            data.const_nums.const_12,
            data.const_nums.const_13,
            data.const_nums.const_14,
            data.const_nums.const_15,
        ];
        let h_indices = [
            data.const_nums.const_0,
            data.const_nums.const_1,
            data.const_nums.const_2,
            data.const_nums.const_3,
            data.const_nums.const_4,
            data.const_nums.const_5,
            data.const_nums.const_6,
            data.const_nums.const_7,
        ];
        for (workspace_i, (v_i, h_i)) in v_indices.iter().zip(h_indices.iter()).enumerate() {
            let mut v_value = self.load(
                &data.memory.v_final.get_at(*v_i),
                &Time::from_element(write_ts),
            );
            v_value = self.xor(h_workspace_2.get(workspace_i), v_value);
            self.store(
                &data.memory.h.get_at(*h_i),
                v_value,
                &Time::from_element(write_ts),
                None,
            );
        }
    }

    pub fn blake2b_mix(
        &mut self,
        v_a: &U64Register,
        v_b: &U64Register,
        v_c: &U64Register,
        v_d: &U64Register,
        x: &U64Register,
        y: &U64Register,
    ) -> (U64Register, U64Register, U64Register, U64Register) {
        let mut v_a_inter = self.add(*v_a, *v_b);
        v_a_inter = self.add(v_a_inter, *x);

        let mut v_d_inter = self.xor(*v_d, *v_a);
        v_d_inter = self.rotate_right(v_d_inter, 32);

        let mut v_c_inter = self.add(*v_c, v_d_inter);

        let mut v_b_inter = self.xor(*v_b, v_c_inter);
        v_b_inter = self.rotate_right(v_b_inter, 24);

        v_a_inter = self.add(v_a_inter, v_b_inter);
        v_a_inter = self.add(v_a_inter, *y);

        v_d_inter = self.xor(v_d_inter, v_a_inter);
        v_d_inter = self.rotate_right(v_d_inter, 16);

        v_c_inter = self.add(v_c_inter, v_d_inter);

        v_b_inter = self.xor(v_b_inter, v_c_inter);
        v_b_inter = self.rotate_right(v_b_inter, 63);

        (v_a_inter, v_b_inter, v_c_inter, v_d_inter)
    }
}

/*
#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::hash::sha::sha256::SHA256Gadget;
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::util::u32_from_le_field_bytes;
    use crate::machine::hash::sha::sha256::util::SHA256Util;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SHAPreprocessingTest;

    impl AirParameters for SHAPreprocessingTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 201;
        const EXTENDED_COLUMNS: usize = 120;
    }

    #[test]
    fn test_sha256_preprocessing() {
        type L = SHAPreprocessingTest;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha256_preprocessing", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let num_rounds = 1 << 3;
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<U32Register>(16))
            .collect::<Vec<_>>();
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let data = builder.sha256_data(&padded_chunks, &end_bits);

        let w_i = builder.sha256_preprocessing(&data);

        // Dummy reads and writes to make the bus argument work.
        let _ = builder.load(
            &data.memory.round_constants.get_at(data.trace.index),
            &Time::zero(),
        );
        let _ = builder.load(
            &data.memory.end_bit.get_at(data.trace.process_id),
            &Time::zero(),
        );

        let num_rows = 64 * num_rounds;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let msg = b"abc";

        let padded_messages = (0..num_rounds).map(|_| SHA256Gadget::pad(msg));
        let mut expected_w = Vec::new();

        for (message, register) in padded_messages.zip_eq(data.public.padded_chunks.iter()) {
            let padded_msg = message
                .chunks_exact(4)
                .map(|slice| u32::from_be_bytes(slice.try_into().unwrap()))
                .map(u32_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            let pre_processed = SHA256Gadget::process_inputs(&message);

            expected_w.push(pre_processed);
        }

        for end_bit in data.public.end_bits.iter() {
            writer.write(&end_bit, &GoldilocksField::ONE, 0);
        }

        writer.write_global_instructions(&stark.air_data);
        (0..num_rounds).for_each(|r| {
            for k in 0..64 {
                let i = r * 64 + k;
                writer.write_row_instructions(&stark.air_data, i);
            }
        });

        for (r, exp_w) in expected_w.iter().enumerate() {
            for (j, exp) in exp_w.iter().enumerate() {
                let w_i_value = u32::from_le_bytes(
                    writer
                        .read(&w_i, 64 * r + j)
                        .map(|x| x.as_canonical_u64() as u8),
                );
                assert_eq!(w_i_value, *exp);
            }
        }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let data = recursive_builder.build::<Config>();

        let mut pw = PartialWitness::new();

        pw.set_target_arr(&public_input, &public);
        stark.set_proof_target(&mut pw, &proof_target, proof);

        let rec_proof = data.prove(pw).unwrap();
        data.verify(rec_proof).unwrap();

        timing.print();
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SHA256Test;

    impl AirParameters for SHA256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 600;
        const EXTENDED_COLUMNS: usize = 342;
    }

    #[test]
    fn test_sha256_byte_stark() {
        type L = SHA256Test;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha256_processing", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let num_rounds = 1 << 3;
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<U32Register>(16))
            .collect::<Vec<_>>();
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let hash_state = builder.sha256(&padded_chunks, &end_bits);

        let num_rows = 64 * num_rounds;
        let stark = builder.build::<C, 2>(num_rows);

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let rec_data = recursive_builder.build::<Config>();

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let msg = b"abc";
        let expected_digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let padded_messages = (0..num_rounds).map(|_| SHA256Gadget::pad(msg));

        for ((message, register), h_arr) in padded_messages
            .zip_eq(padded_chunks.iter())
            .zip(hash_state.iter())
        {
            let padded_msg = message
                .chunks_exact(4)
                .map(|slice| u32::from_be_bytes(slice.try_into().unwrap()))
                .map(u32_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            let pre_processed = SHA256Gadget::process_inputs(&message);
            let state = SHA256Gadget::compress_round(IV, &pre_processed, ROUND_CONSTANTS)
                .map(u32_to_le_field_bytes);
            writer.write_slice(h_arr, &state.concat(), 0);
        }
        for end_bit in end_bits.iter() {
            writer.write(&end_bit, &GoldilocksField::ONE, 0);
        }

        timed!(timing, "write input", {
            writer.write_global_instructions(&stark.air_data);
            for i in 0..num_rows {
                writer.write_row_instructions(&stark.air_data, i);
            }
        });

        // Compare expected digests with the trace values.
        let expected_digest = SHA256Util::decode(expected_digest);
        for state in hash_state.iter() {
            let digest = writer
                .read_array::<_, 8>(&state.as_array(), 0)
                .map(|x| u32_from_le_field_bytes(&x));
            assert_eq!(digest, expected_digest);
        }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = timed!(
            timing,
            "generate stark proof",
            stark.prove(&trace, &public, &mut timing).unwrap()
        );

        stark.verify(proof.clone(), &public).unwrap();

        let mut pw = PartialWitness::new();

        pw.set_target_arr(&public_input, &public);
        stark.set_proof_target(&mut pw, &proof_target, proof);

        let rec_proof = timed!(
            timing,
            "generate recursive proof",
            rec_data.prove(pw).unwrap()
        );
        rec_data.verify(rec_proof).unwrap();

        timing.print();
    }
}
*/
