use core::fmt::Debug;

use log::debug;
use num::Zero;
use plonky2::util::log2_ceil;
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::data::SHAData;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::machine::builder::Builder;
use crate::machine::hash::sha::data::{SHAMemory, SHAPublicData, SHATraceData};
use crate::machine::hash::{HashDigest, HashIntConversion, HashPureInteger};
use crate::math::prelude::*;

const DUMMY_INDEX: u64 = i32::MAX as u64;

/// Pure SHA algorithm implementation.
///
/// An interface for the SHA algorithm as a Rust function operating on numerical values.
pub trait SHAPure<const CYCLE_LENGTH: usize>:
    Debug + Clone + 'static + Serialize + DeserializeOwned + Send + Sync + HashPureInteger
{
    const INITIAL_HASH: [Self::Integer; 8];
    const ROUND_CONSTANTS: [Self::Integer; CYCLE_LENGTH];

    /// Pad a byte message to a vector of `Self::Integer` values.
    fn pad(msg: &[u8]) -> Vec<Self::Integer>;

    /// Pre-process a chunk of `Self::Integer` values.
    fn pre_process(chunk: &[Self::Integer]) -> [Self::Integer; CYCLE_LENGTH];

    /// Process a chunk of `Self::Integer` values.
    fn process(hash: [Self::Integer; 8], w: &[Self::Integer; CYCLE_LENGTH]) -> [Self::Integer; 8];

    /// Decode a digest encoded as a string to a vector of `Self::Integer` values.
    fn decode(digest: &str) -> [Self::Integer; 8];
}

/// SHA algorithm AIR implementation.
///
/// An interface for the SHA algorithm as an AIR.
pub trait SHAir<B: Builder, const CYCLE_LENGTH: usize>:
    SHAPure<CYCLE_LENGTH> + HashIntConversion<B> + HashDigest<B>
{
    type StateVariable: Register + Into<ArrayRegister<Self::IntRegister>>;
    type StatePointer;

    /// The clock register, whose value equals the current row.
    fn clk(builder: &mut B) -> ElementRegister;

    /// End bits for a 16-cycle and a CYCLE_LENGTH-cycle.
    ///
    /// This function should return two bits `(cycle_16_end_bit, cycle_end_bit)` such that
    ///     -  `cycle_16_end_bit` is `1` at the end of every 16 cycles and `0` otherwise.
    ///     - `cycle_end_bit` is `1` at the end of every CYCLE_LENGTH cycles and `0` otherwise.
    fn cycles_end_bits(builder: &mut B) -> (BitRegister, BitRegister);

    /// Given the elements `w[i-15]`, `w[i-2]`, `w[i-16]`, `w[i-7]`, compute `w[i]`.
    fn preprocessing_step(
        builder: &mut B,
        w_i_minus_15: Self::IntRegister,
        w_i_minus_2: Self::IntRegister,
        w_i_mimus_16: Self::IntRegister,
        w_i_mimus_7: Self::IntRegister,
    ) -> Self::IntRegister;

    fn processing_step(
        builder: &mut B,
        vars: ArrayRegister<Self::IntRegister>,
        w_i: Self::IntRegister,
        round_constant: Self::IntRegister,
    ) -> Vec<Self::IntRegister>;

    fn load_state(
        builder: &mut B,
        hash_state_public: &[Self::StateVariable],
        digest_indices: ArrayRegister<ElementRegister>,
    ) -> Self::StatePointer;

    fn store_state(
        builder: &mut B,
        state_ptr: &Self::StatePointer,
        state_next: Self::StateVariable,
        time: &Time<B::Field>,
        flag: Option<ElementRegister>,
    );

    fn absorb(
        builder: &mut B,
        state: ArrayRegister<Self::IntRegister>,
        vars_next: &[Self::IntRegister],
    ) -> Self::StateVariable;

    fn sha(
        builder: &mut B,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: ArrayRegister<ElementRegister>,
    ) -> Vec<Self::StateVariable> {
        let data = Self::data(
            builder,
            padded_chunks,
            end_bits,
            digest_bits,
            digest_indices,
        );
        let w_i = Self::preprocessing(builder, &data);
        Self::processing(builder, w_i, &data)
    }

    fn data(
        builder: &mut B,
        padded_chunks: &[ArrayRegister<Self::IntRegister>],
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: ArrayRegister<ElementRegister>,
    ) -> SHAData<Self::IntRegister, CYCLE_LENGTH> {
        assert_eq!(padded_chunks.len(), end_bits.len());
        let num_real_rounds = padded_chunks.len();
        debug!(
            "AIR degree before padding: {}",
            num_real_rounds * CYCLE_LENGTH
        );
        let degree_log = log2_ceil(num_real_rounds * CYCLE_LENGTH);
        assert!(degree_log < 31, "AIR degree is too large");
        debug!("AIR degree after padding: {}", 1 << degree_log);
        let num_dummy_rounds = (1 << degree_log) / CYCLE_LENGTH + 1 - num_real_rounds;
        // Keep track of the last round length to know how many dummy reads to add.
        let length_last_round = (1 << degree_log) % CYCLE_LENGTH;
        let num_rounds = num_real_rounds + num_dummy_rounds;

        // Convert the number of rounds to a field element.
        let num_round_element = builder.constant(&B::Field::from_canonical_usize(num_rounds));
        let num_round_minus_one = builder.constant(&B::Field::from_canonical_usize(num_rounds - 1));

        // Initialize the initial hash and set it to the constant value.
        let initial_hash = builder
            .constant_array::<Self::IntRegister>(&Self::INITIAL_HASH.map(Self::int_to_field_value));

        // Initialize the round constants and set them to the constant value.
        let round_constant_values = builder.constant_array::<Self::IntRegister>(
            &Self::ROUND_CONSTANTS.map(Self::int_to_field_value),
        );

        // Store the round constants in a slice to be able to load them in the trace.
        let round_constants = builder.uninit_slice();

        for i in 0..length_last_round {
            builder.store(
                &round_constants.get(i),
                round_constant_values.get(i),
                &Time::zero(),
                Some(num_round_element),
                None,
                None,
            );
        }

        for i in length_last_round..CYCLE_LENGTH {
            builder.store(
                &round_constants.get(i),
                round_constant_values.get(i),
                &Time::zero(),
                Some(num_round_minus_one),
                None,
                None,
            );
        }

        // Initialize shift read multiplicities with zeros.
        let mut shift_read_mult = [B::Field::ZERO; CYCLE_LENGTH];
        let read_len = CYCLE_LENGTH - 16;
        // Add multiplicities for reading the elements w[i-15].
        for mult in shift_read_mult.iter_mut().skip(16 - 15).take(read_len) {
            *mult += B::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-2].
        for mult in shift_read_mult.iter_mut().skip(16 - 2).take(read_len) {
            *mult += B::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-16].
        for mult in shift_read_mult.iter_mut().take(read_len) {
            *mult += B::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-7].
        for mult in shift_read_mult.iter_mut().skip(16 - 7).take(read_len) {
            *mult += B::Field::ONE;
        }

        let shift_read_values = builder.constant_array::<ElementRegister>(&shift_read_mult);

        let shift_read_mult = builder.uninit_slice();

        for i in 0..length_last_round {
            builder.store(
                &shift_read_mult.get(i),
                shift_read_values.get(i),
                &Time::zero(),
                Some(num_round_element),
                None,
                None,
            );
        }
        for i in length_last_round..CYCLE_LENGTH {
            builder.store(
                &shift_read_mult.get(i),
                shift_read_values.get(i),
                &Time::zero(),
                Some(num_round_minus_one),
                None,
                None,
            );
        }

        let w = builder.uninit_slice();
        let dummy_entry =
            builder.constant::<Self::IntRegister>(&Self::int_to_field_value(Self::Integer::zero()));

        assert!(DUMMY_INDEX < B::Field::order());
        let dummy_index = builder.constant(&B::Field::from_canonical_u64(DUMMY_INDEX));

        let num_dummy_reads = builder.constant::<ElementRegister>(&B::Field::from_canonical_usize(
            num_real_rounds * (16 * 4 + read_len)
                + (num_dummy_rounds - 1) * CYCLE_LENGTH * 5
                + length_last_round * 5,
        ));

        for (i, padded_chunk) in padded_chunks.iter().enumerate() {
            for (j, word) in padded_chunk.iter().enumerate().take(16) {
                builder.store(
                    &w.get(CYCLE_LENGTH * i + j),
                    word,
                    &Time::zero(),
                    None,
                    None,
                    None,
                );
            }
        }

        builder.store(
            &w.get(DUMMY_INDEX as usize),
            dummy_entry,
            &Time::zero(),
            Some(num_dummy_reads),
            None,
            None,
        );

        let (cycle_16_end_bit, cycle_end_bit) = Self::cycles_end_bits(builder);

        // `process_id` is a register is computed by counting the number of cycles. We do this by
        // setting `process_id` to be the cumulative sum of the `end_bit` of each cycle.
        let process_id = builder.process_id(CYCLE_LENGTH, cycle_end_bit);
        // The array index register can be computed as `clock - process_id * CYCLE_LENGTH`.
        let clk = Self::clk(builder);
        let index = builder.expression(
            clk.expr() - process_id.expr() * B::Field::from_canonical_usize(CYCLE_LENGTH),
        );

        // Preprocessing happens in steps 16..CYCLE_LENGTH of each CYCLE_LENGTH-cycle. We compute
        // this register by having an accumnumator `is_preprocessing` so that:
        //    - `is_preprocessing` becomes `0` at the beginning of each CYCLE_LENGTH cycle.
        //    - `is_preprocessing` becomes `1` at the end of every 16 cycle unless this coincides
        //       with the end of a CYCLE_LENGTH-cycle.
        //    - otherwise, `is_preprocessing` remains the same.
        let is_preprocessing = builder.alloc::<BitRegister>();
        builder.set_to_expression_first_row(&is_preprocessing, B::Field::ZERO.into());
        builder.set_to_expression_transition(
            &is_preprocessing.next(),
            cycle_end_bit.not_expr()
                * (cycle_16_end_bit.expr() + cycle_16_end_bit.not_expr() * is_preprocessing.expr()),
        );

        // Allocate end_bits for public input.
        let one = builder.constant(&B::Field::ONE);
        let zero = builder.constant(&B::Field::ZERO);
        let reg_cycle_length = builder.constant(&B::Field::from_canonical_usize(CYCLE_LENGTH));
        let reg_last_length = builder.constant(&B::Field::from_canonical_usize(length_last_round));
        let end_bit = builder.uninit_slice();
        for (i, end_bit_val) in end_bits.iter().enumerate() {
            builder.store(
                &end_bit.get(i),
                end_bit_val,
                &Time::zero(),
                Some(reg_cycle_length),
                None,
                None,
            );
        }
        for i in num_real_rounds..num_rounds - 1 {
            builder.store(
                &end_bit.get(i),
                zero,
                &Time::zero(),
                Some(reg_cycle_length),
                None,
                None,
            );
        }
        builder.store(
            &end_bit.get(num_rounds - 1),
            zero,
            &Time::zero(),
            Some(reg_last_length),
            None,
            None,
        );
        let digest_bit = builder.uninit_slice();
        for (i, digest_bit_val) in digest_bits.iter().enumerate() {
            builder.store(
                &digest_bit.get(i),
                digest_bit_val,
                &Time::zero(),
                Some(reg_cycle_length),
                None,
                None,
            );
        }
        for i in num_real_rounds..num_rounds - 1 {
            builder.store(
                &digest_bit.get(i),
                zero,
                &Time::zero(),
                Some(reg_cycle_length),
                None,
                None,
            );
        }
        builder.store(
            &digest_bit.get(num_rounds - 1),
            zero,
            &Time::zero(),
            Some(reg_last_length),
            None,
            None,
        );

        // Initialize a bit slice to commit to `is_dummy` bits.
        let is_dummy_slice = builder.uninit_slice();

        for i in 0..num_real_rounds {
            builder.store(
                &is_dummy_slice.get(i),
                zero,
                &Time::zero(),
                Some(reg_cycle_length),
                None,
                None,
            );
        }
        for i in num_real_rounds..num_rounds - 1 {
            builder.store(
                &is_dummy_slice.get(i),
                one,
                &Time::zero(),
                Some(reg_cycle_length),
                None,
                None,
            );
        }
        let last_round_reg = builder.constant(&B::Field::from_canonical_usize(length_last_round));
        builder.store(
            &is_dummy_slice.get(num_rounds - 1),
            one,
            &Time::zero(),
            Some(last_round_reg),
            None,
            None,
        );
        let is_dummy = builder.load(
            &is_dummy_slice.get_at(process_id),
            &Time::zero(),
            None,
            None,
        );

        let public = SHAPublicData {
            initial_hash,
            padded_chunks: padded_chunks.to_vec(),
            digest_indices,
        };

        let trace = SHATraceData {
            is_preprocessing,
            process_id,
            cycle_end_bit,
            index,
            is_dummy,
        };

        let memory = SHAMemory {
            round_constants,
            w,
            shift_read_mult,
            end_bit,
            digest_bit,
            dummy_index,
        };
        SHAData {
            public,
            trace,
            memory,
            degree: 1 << degree_log,
        }
    }

    fn preprocessing(
        builder: &mut B,
        data: &SHAData<Self::IntRegister, CYCLE_LENGTH>,
    ) -> Self::IntRegister {
        let w = &data.memory.w;
        let dummy_index = data.memory.dummy_index;
        let is_preprocessing = data.trace.is_preprocessing;
        let shift_read_mult = &data.memory.shift_read_mult;
        let is_dummy = data.trace.is_dummy;
        let clk = Self::clk(builder);

        let time = Time::zero();

        let shifted_index = |i: u32, builder: &mut B| {
            builder.expression(
                is_dummy.expr() * dummy_index.expr()
                    + is_dummy.not_expr()
                        * (is_preprocessing.expr()
                            * (clk.expr() - B::Field::from_canonical_u32(i))
                            + is_preprocessing.not_expr() * dummy_index.expr()),
            )
        };

        let i_m_15 = shifted_index(15, builder);
        let w_i_minus_15 = builder.load(&w.get_at(i_m_15), &time, None, None);
        let i_m_2 = shifted_index(2, builder);
        let w_i_minus_2 = builder.load(&w.get_at(i_m_2), &time, None, None);

        let i_m_16 = shifted_index(16, builder);
        let w_i_mimus_16 = builder.load(&w.get_at(i_m_16), &time, None, None);
        let i_m_7 = shifted_index(7, builder);
        let w_i_mimus_7 = builder.load(&w.get_at(i_m_7), &time, None, None);

        let w_i_pre_process = Self::preprocessing_step(
            builder,
            w_i_minus_15,
            w_i_minus_2,
            w_i_mimus_16,
            w_i_mimus_7,
        );

        let mut i_idx = builder.select(is_preprocessing, &dummy_index, &clk);
        i_idx = builder.select(is_dummy, &dummy_index, &i_idx);
        let w_i_read = builder.load(&w.get_at(i_idx), &time, None, None);

        let w_i = builder.select(is_preprocessing, &w_i_pre_process, &w_i_read);

        let mut reading_mult = builder.load(
            &shift_read_mult.get_at(data.trace.index),
            &Time::zero(),
            None,
            None,
        );
        reading_mult = builder.expression(reading_mult.expr() * is_dummy.not_expr());
        builder.store(&w.get_at(clk), w_i, &time, Some(reading_mult), None, None);

        w_i
    }

    fn processing(
        builder: &mut B,
        w_i: Self::IntRegister,
        data: &SHAData<Self::IntRegister, CYCLE_LENGTH>,
    ) -> Vec<Self::StateVariable> {
        let num_digests = data.public.digest_indices.len();
        let hash_state_public = (0..num_digests)
            .map(|_| builder.alloc_public::<Self::StateVariable>())
            .collect::<Vec<_>>();
        let state_ptr = Self::load_state(builder, &hash_state_public, data.public.digest_indices);

        let index = data.trace.index;
        let initial_hash = data.public.initial_hash;
        let cycle_end_bit = data.trace.cycle_end_bit;

        let round_constant = builder.load(
            &data.memory.round_constants.get_at(index),
            &Time::zero(),
            None,
            None,
        );

        // Initialize working variables
        let state = builder.alloc_array::<Self::IntRegister>(8);
        for (h, h_init) in state.iter().zip(initial_hash.iter()) {
            builder.set_to_expression_first_row(&h, h_init.expr());
        }
        // Initialize working variables and set them to the inital hash in the first row.
        let vars = builder.alloc_array::<Self::IntRegister>(8);
        for (v, h_init) in vars.iter().zip(initial_hash.iter()) {
            builder.set_to_expression_first_row(&v, h_init.expr());
        }

        let vars_next = Self::processing_step(builder, vars, w_i, round_constant);

        let state_next = Self::absorb(builder, state, &vars_next);

        // Store the new state values
        let process_id = data.trace.process_id;
        let is_dummy = data.trace.is_dummy;
        let digest_bit = builder.load(
            &data.memory.digest_bit.get_at(data.trace.process_id),
            &Time::zero(),
            None,
            None,
        );
        let flag = Some(
            builder.expression(cycle_end_bit.expr() * is_dummy.not_expr() * digest_bit.expr()),
        );
        Self::store_state(
            builder,
            &state_ptr,
            state_next,
            &Time::from_element(process_id),
            flag,
        );

        // Set the next row of working variables.
        let end_bit = builder.load(
            &data.memory.end_bit.get_at(data.trace.process_id),
            &Time::zero(),
            None,
            None,
        );
        let bit = cycle_end_bit;
        let state_next_arr: ArrayRegister<Self::IntRegister> = state_next.into();
        for ((((var, h), init), var_next), h_next) in vars
            .iter()
            .zip(state.iter())
            .zip(initial_hash.iter())
            .zip(vars_next.iter())
            .zip(state_next_arr.iter())
        {
            builder.set_to_expression_transition(
                &var.next(),
                var_next.expr() * bit.not_expr()
                    + (h_next.expr() * end_bit.not_expr() + init.expr() * end_bit.expr())
                        * bit.expr(),
            );
            builder.set_to_expression_transition(
                &h.next(),
                h.expr() * bit.not_expr()
                    + (h_next.expr() * end_bit.not_expr() + init.expr() * end_bit.expr())
                        * bit.expr(),
            );
        }

        hash_state_public
    }
}
