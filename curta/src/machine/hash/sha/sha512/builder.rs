use log::debug;

use super::data::{SHA512Data, SHA512Memory, SHA512PublicData, SHA512TraceData};
use super::register::SHA512DigestRegister;
use super::{INITIAL_HASH, ROUND_CONSTANTS};
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
use crate::math::prelude::*;

impl<L: AirParameters> BytesBuilder<L>
where
    L::Instruction: UintInstructions,
{
    pub fn sha512(
        &mut self,
        padded_chunks: &[ArrayRegister<U64Register>],
        end_bits: &ArrayRegister<BitRegister>,
    ) -> Vec<SHA512DigestRegister> {
        let data = self.sha512_data(padded_chunks, end_bits);
        let w_i = self.sha512_preprocessing(&data);
        self.sha512_processing(w_i, &data)
    }

    pub fn sha512_data(
        &mut self,
        padded_chunks: &[ArrayRegister<U64Register>],
        end_bits: &ArrayRegister<BitRegister>,
    ) -> SHA512Data {
        assert_eq!(padded_chunks.len(), end_bits.len());
        let num_real_rounds = padded_chunks.len();
        debug!("AIR degree before padding: {}", num_real_rounds * 80);
        let degree = (num_real_rounds * 80).ilog2() + 1;
        debug!("AIR degree after passing: {}", 1 << degree);
        let num_dummy_rounds = (1 << degree) / 80 + 1 - num_real_rounds;
        // Keep track of the last round length to know how many dummy reads to add.
        let length_last_round = (1 << degree) % 80;
        let num_rounds = num_real_rounds + num_dummy_rounds;
        debug!("num rounds: {}", num_rounds);
        debug!("num real rounds: {}", num_real_rounds);

        // Convert the number of rounds to a field element.
        let num_round_element = self.constant(&L::Field::from_canonical_usize(num_rounds));
        let num_round_minus_one = self.constant(&L::Field::from_canonical_usize(num_rounds - 1));

        // Initialize the initial hash and set it to the constant value.
        let initial_hash =
            self.constant_array::<U64Register>(&INITIAL_HASH.map(u64_to_le_field_bytes));

        // Initialize the round constants and set them to the constant value.
        let round_constant_values =
            self.constant_array::<U64Register>(&ROUND_CONSTANTS.map(u64_to_le_field_bytes));

        // Store the round constants in a slice to be able to load them in the trace.
        let round_constants = self.uninit_slice();

        for i in 0..length_last_round {
            self.store(
                &round_constants.get(i),
                round_constant_values.get(i),
                &Time::zero(),
                Some(num_round_element),
            );
        }

        for i in length_last_round..80 {
            self.store(
                &round_constants.get(i),
                round_constant_values.get(i),
                &Time::zero(),
                Some(num_round_minus_one),
            );
        }

        // Initialize shift read multiplicities with zeros.
        let mut shift_read_mult = [L::Field::ZERO; 80];
        // Add multiplicities for reading the elements w[i-15].
        for mult in shift_read_mult.iter_mut().skip(1).take(64) {
            *mult += L::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-2].
        for mult in shift_read_mult.iter_mut().skip(14).take(64) {
            *mult += L::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-16].
        for mult in shift_read_mult.iter_mut().take(64) {
            *mult += L::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-7].
        for mult in shift_read_mult.iter_mut().skip(16 - 7).take(64) {
            *mult += L::Field::ONE;
        }

        let shift_read_values = self.constant_array::<ElementRegister>(&shift_read_mult);

        let shift_read_mult = self.uninit_slice();

        for i in 0..length_last_round {
            self.store(
                &shift_read_mult.get(i),
                shift_read_values.get(i),
                &Time::zero(),
                Some(num_round_element),
            );
        }
        for i in length_last_round..80 {
            self.store(
                &shift_read_mult.get(i),
                shift_read_values.get(i),
                &Time::zero(),
                Some(num_round_minus_one),
            );
        }

        let w = self.uninit_slice();
        let dummy_entry = self.constant::<U64Register>(&[L::Field::ZERO; 8]);

        let dummy_index = self.constant(&L::Field::from_canonical_u8(80));

        let num_dummy_reads =
            self.constant::<ElementRegister>(&L::Field::from_canonical_usize(16 * 4 + 64));

        for (i, padded_chunk) in padded_chunks.iter().enumerate() {
            for (j, word) in padded_chunk.iter().enumerate().take(16) {
                self.store(&w.get(j), word, &Time::constant(i), None);
            }
            self.store(
                &w.get(80),
                dummy_entry,
                &Time::constant(i),
                Some(num_dummy_reads),
            );
        }

        let num_reads_dummy_full =
            self.constant::<ElementRegister>(&L::Field::from_canonical_usize(80 * 5));

        let num_reads_dummy_last = self
            .constant::<ElementRegister>(&L::Field::from_canonical_usize(length_last_round * 5));
        for i in num_real_rounds..num_rounds - 1 {
            self.store(
                &w.get(80),
                dummy_entry,
                &Time::constant(i),
                Some(num_reads_dummy_full),
            );
        }

        self.store(
            &w.get(80),
            dummy_entry,
            &Time::constant(num_rounds - 1),
            Some(num_reads_dummy_last),
        );

        // Initialize cycles to generate process id and is_processing flag.
        let cycle_16 = self.cycle(4);
        let cycle_80_end_bit = {
            let loop_5 = self.api.loop_instr(5);
            let five_end_bit = loop_5.get_iteration_reg(4);
            self.mul(five_end_bit, cycle_16.end_bit)
        };

        // `process_id` is a register is computed by counting the number of cycles. We do this by
        // setting `process_id` to be the cumulatibe sum of the `end_bit` of each cycle.
        let process_id = self.alloc::<ElementRegister>();
        self.set_to_expression_first_row(&process_id, L::Field::ZERO.into());
        self.set_to_expression_transition(
            &process_id.next(),
            process_id.expr() + cycle_80_end_bit.expr(),
        );
        // The array index register can be computed as `clock - process_id * 80`.
        let index =
            self.expression(self.clk.expr() - process_id.expr() * L::Field::from_canonical_u32(80));

        // Preprocessing happens in steps 16..80 of each 80-cycle. We compute this register by
        // having an accumnumator so that:
        //    - `is_preprocessing` becomes `0` at the beginning of each 64 cycle.
        //    - `is_preprocessing` becomes `1` at the end of every 16 cycle unless this coincides
        //       with the end of a 80-cycle.
        //    - otherwise, `is_preprocessing` remains the same.
        let is_preprocessing = self.alloc::<BitRegister>();
        self.set_to_expression_first_row(&is_preprocessing, L::Field::ZERO.into());
        self.set_to_expression_transition(
            &is_preprocessing.next(),
            cycle_80_end_bit.not_expr()
                * (cycle_16.end_bit.expr() + cycle_16.end_bit.not_expr() * is_preprocessing.expr()),
        );

        // Allocate end_bits for public input.
        let one = self.constant(&L::Field::ONE);
        let zero = self.constant(&L::Field::ZERO);
        let reg_80 = self.constant(&L::Field::from_canonical_u8(80));
        let reg_last_length = self.constant(&L::Field::from_canonical_usize(length_last_round));
        let end_bit = self.uninit_slice();
        for (i, end_bit_val) in end_bits.iter().enumerate() {
            self.store(&end_bit.get(i), end_bit_val, &Time::zero(), Some(reg_80));
        }
        for i in num_real_rounds..num_rounds - 1 {
            self.store(&end_bit.get(i), zero, &Time::zero(), Some(reg_80));
        }
        self.store(
            &end_bit.get(num_rounds - 1),
            zero,
            &Time::zero(),
            Some(reg_last_length),
        );

        // Initialize a bit slice to commit to `is_dummy` bits.
        let is_dummy_slice = self.uninit_slice();

        for i in 0..num_real_rounds {
            self.store(&is_dummy_slice.get(i), zero, &Time::zero(), Some(reg_80));
        }
        for i in num_real_rounds..num_rounds - 1 {
            self.store(&is_dummy_slice.get(i), one, &Time::zero(), Some(reg_80));
        }
        debug!("length last round: {}", length_last_round);
        let last_round_reg = self.constant(&L::Field::from_canonical_usize(length_last_round));
        self.store(
            &is_dummy_slice.get(num_rounds - 1),
            one,
            &Time::zero(),
            Some(last_round_reg),
        );
        let is_dummy = self.load(&is_dummy_slice.get_at(process_id), &Time::zero());
        debug!("is_dummy: {:?}", *is_dummy.register());

        let public = SHA512PublicData {
            initial_hash,
            padded_chunks: padded_chunks.to_vec(),
            end_bits: *end_bits,
        };

        let trace = SHA512TraceData {
            is_preprocessing,
            process_id,
            cycle_80_end_bit,
            index,
            is_dummy,
        };

        let memory = SHA512Memory {
            round_constants,
            w,
            shift_read_mult,
            end_bit,
            dummy_index,
        };
        SHA512Data {
            public,
            trace,
            memory,
            num_chunks: num_rounds,
            degree: 1 << degree,
        }
    }

    pub fn sha512_preprocessing(&mut self, data: &SHA512Data) -> U64Register {
        let w = &data.memory.w;
        let index = data.trace.index;
        let dummy_index = data.memory.dummy_index;
        let process_id = data.trace.process_id;
        let is_preprocessing = data.trace.is_preprocessing;
        let shift_read_mult = &data.memory.shift_read_mult;
        let is_dummy = data.trace.is_dummy;

        let time = Time::from_element(process_id);

        let shifted_index = |i: u32, builder: &mut BytesBuilder<L>| {
            builder.expression(
                is_dummy.expr() * dummy_index.expr()
                    + is_dummy.not_expr()
                        * (is_preprocessing.expr()
                            * (index.expr() - L::Field::from_canonical_u32(i))
                            + is_preprocessing.not_expr() * dummy_index.expr()),
            )
        };

        // Calculate the value:
        // s_0 = w_i_minus_15.rotate_right(1) ^ w_i_minus_15.rotate_right(8) ^ (w_i_minus_15 >> 7)
        let i_m_15 = shifted_index(15, self);
        let w_i_minus_15 = self.load(&w.get_at(i_m_15), &time);
        debug!("w_i_minus_15: {:?}", *w_i_minus_15.register());
        let w_i_minus_15_rotate_1 = self.rotate_right(w_i_minus_15, 1);
        let w_i_minus_15_rotate_8 = self.rotate_right(w_i_minus_15, 8);
        let w_i_minus_15_shr_7 = self.shr(w_i_minus_15, 7);

        let mut s_0 = self.xor(&w_i_minus_15_rotate_1, &w_i_minus_15_rotate_8);
        s_0 = self.xor(&s_0, &w_i_minus_15_shr_7);

        // Calculate the value:
        // s_1 = w_i_minus_2.rotate_right(19) ^ w_i_minus_2.rotate_right(61) ^ (w_i_minus_2 >> 6)
        let i_m_2 = shifted_index(2, self);
        let w_i_minus_2 = self.load(&w.get_at(i_m_2), &time);
        debug!("w_i_minus_2: {:?}", *w_i_minus_2.register());
        let w_i_minus_2_rotate_19 = self.rotate_right(w_i_minus_2, 19);
        let w_i_minus_2_rotate_61 = self.rotate_right(w_i_minus_2, 61);
        let w_i_minus_2_shr_6 = self.shr(w_i_minus_2, 6);

        let mut s_1 = self.xor(&w_i_minus_2_rotate_19, &w_i_minus_2_rotate_61);
        s_1 = self.xor(&s_1, &w_i_minus_2_shr_6);

        // Calculate the value:
        // w_i = w_i_minus_16 + s_0 + w_i_minus_7 + s_1
        let i_m_16 = shifted_index(16, self);
        let w_i_mimus_16 = self.load(&w.get_at(i_m_16), &time);
        let i_m_7 = shifted_index(7, self);
        let w_i_mimus_7 = self.load(&w.get_at(i_m_7), &time);
        let mut w_i_pre_process = self.add(w_i_mimus_16, s_0);
        w_i_pre_process = self.add(w_i_pre_process, w_i_mimus_7);
        w_i_pre_process = self.add(w_i_pre_process, s_1);

        let mut i_idx = self.select(is_preprocessing, &dummy_index, &index);
        i_idx = self.select(is_dummy, &dummy_index, &i_idx);
        let w_i_read = self.load(&w.get_at(i_idx), &time);
        debug!("w_i_read: {:?}", *w_i_read.register());

        let w_i = self.select(is_preprocessing, &w_i_pre_process, &w_i_read);

        let mut reading_mult = self.load(&shift_read_mult.get_at(index), &Time::zero());
        reading_mult = self.expression(reading_mult.expr() * is_dummy.not_expr());
        self.store(&w.get_at(index), w_i, &time, Some(reading_mult));

        w_i
    }

    /// The processing step of a sha512 round.
    pub fn sha512_processing(
        &mut self,
        w_i: U64Register,
        data: &SHA512Data,
    ) -> Vec<SHA512DigestRegister> {
        let hash_state_public = (0..data.num_chunks)
            .map(|_| self.alloc_public::<SHA512DigestRegister>())
            .collect::<Vec<_>>();
        let state_ptr = self.uninit_slice();

        for (i, h_slice) in hash_state_public.iter().enumerate() {
            for (j, h) in h_slice.iter().enumerate() {
                self.free(&state_ptr.get(j), h, &Time::constant(i));
            }
        }
        let round_constant = &data.memory.round_constants;
        let index = data.trace.index;
        let initial_hash = data.public.initial_hash;
        let cycle_end_bit = data.trace.cycle_80_end_bit;

        let round_constant = self.load(&round_constant.get_at(index), &Time::zero());

        // Initialize working variables
        let state = self.alloc_array::<U64Register>(8);
        for (h, h_init) in state.iter().zip(initial_hash.iter()) {
            self.set_to_expression_first_row(&h, h_init.expr());
        }
        // Initialize working variables and set them to the inital hash in the first row.
        let vars = self.alloc_array::<U64Register>(8);
        for (v, h_init) in vars.iter().zip(initial_hash.iter()) {
            self.set_to_expression_first_row(&v, h_init.expr());
        }

        let a = vars.get(0);
        let b = vars.get(1);
        let c = vars.get(2);
        let d = vars.get(3);
        let e = vars.get(4);
        let f = vars.get(5);
        let g = vars.get(6);
        let h = vars.get(7);

        // Calculate S1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41).
        let e_rotate_14 = self.rotate_right(e, 14);
        let e_rotate_18 = self.rotate_right(e, 18);
        let e_rotate_41 = self.rotate_right(e, 41);
        let mut sum_1 = self.xor(e_rotate_14, e_rotate_18);
        sum_1 = self.xor(sum_1, e_rotate_41);

        // Calculate ch = (e & f) ^ (!e & g).
        let e_and_f = self.and(&e, &f);
        let not_e = self.not(e);
        let not_e_and_g = self.and(&not_e, &g);
        let ch = self.xor(&e_and_f, &not_e_and_g);

        // Calculate temp_1 = h + sum_1 + ch + round_constant + w.
        let mut temp_1 = self.add(h, sum_1);
        temp_1 = self.add(temp_1, ch);
        temp_1 = self.add(temp_1, round_constant);
        temp_1 = self.add(temp_1, w_i);

        // Calculate S0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22).
        let a_rotate_2 = self.rotate_right(a, 2);
        let a_rotate_13 = self.rotate_right(a, 13);
        let a_rotate_22 = self.rotate_right(a, 22);
        let mut sum_0 = self.xor(a_rotate_2, a_rotate_13);
        sum_0 = self.xor(sum_0, a_rotate_22);

        // Calculate maj = (a & b) ^ (a & c) ^ (b & c);
        let a_and_b = self.and(a, b);
        let a_and_c = self.and(a, c);
        let b_and_c = self.and(b, c);
        let mut maj = self.xor(a_and_b, a_and_c);
        maj = self.xor(maj, b_and_c);

        // Calculate temp_2 = sum_0 + maj.
        let temp_2 = self.add(sum_0, maj);

        // Calculate the next cycle values.
        let mut vars_next = Vec::new();
        let a_next = self.add(temp_1, temp_2);
        let b_next = a;
        let c_next = b;
        let d_next = c;
        let e_next = self.add(d, temp_1);
        let f_next = e;
        let g_next = f;
        let h_next = g;
        vars_next.push(a_next);
        vars_next.push(b_next);
        vars_next.push(c_next);
        vars_next.push(d_next);
        vars_next.push(e_next);
        vars_next.push(f_next);
        vars_next.push(g_next);
        vars_next.push(h_next);

        let state_plus_vars = self.alloc_array(8);
        for ((s, v), res) in state
            .iter()
            .zip(vars_next.iter())
            .zip(state_plus_vars.iter())
        {
            let carry = self.alloc();
            self.api
                .set_add_u64(&s, v, &None, &res, &carry, &mut self.operations)
        }

        // Store the new state values
        let flag = Some(cycle_end_bit.as_element());
        let process_id = data.trace.process_id;
        for i in 0..4 {
            let val = U64Register::from_limbs(&state_plus_vars.get_subarray(i * 2..i * 2 + 2));
            self.store(
                &state_ptr.get(i),
                val,
                &Time::from_element(process_id),
                flag,
            );
        }

        // Set the next row of working variables.
        let end_bit = self.load(
            &data.memory.end_bit.get_at(data.trace.process_id),
            &Time::zero(),
        );
        let bit = cycle_end_bit;
        for ((((var, h), init), var_next), h_next) in vars
            .iter()
            .zip(state.iter())
            .zip(initial_hash.iter())
            .zip(vars_next.iter())
            .zip(state_plus_vars.iter())
        {
            self.set_to_expression_transition(
                &var.next(),
                var_next.expr() * bit.not_expr()
                    + (h_next.expr() * end_bit.not_expr() + init.expr() * end_bit.expr())
                        * bit.expr(),
            );
            self.set_to_expression_transition(
                &h.next(),
                h.expr() * bit.not_expr()
                    + (h_next.expr() * end_bit.not_expr() + init.expr() * end_bit.expr())
                        * bit.expr(),
            );
        }

        hash_state_public
    }
}

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
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::machine::hash::sha::sha512::util::SHA512Util;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SHAPreprocessingTest;

    impl AirParameters for SHAPreprocessingTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 391;
        const EXTENDED_COLUMNS: usize = 207;
    }

    #[test]
    fn test_sha512_preprocessing() {
        type L = SHAPreprocessingTest;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha512_preprocessing", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let num_rounds = 2;
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<U64Register>(16))
            .collect::<Vec<_>>();
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let data = builder.sha512_data(&padded_chunks, &end_bits);

        let w_i = builder.sha512_preprocessing(&data);

        // Dummy reads and writes to make the bus argument work.
        let round_constant = builder.load(
            &data.memory.round_constants.get_at(data.trace.index),
            &Time::zero(),
        );
        debug!("round_constant: {:?}", *round_constant.register());
        let _ = builder.load(
            &data.memory.end_bit.get_at(data.trace.process_id),
            &Time::zero(),
        );

        let num_rows = data.degree;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let msg = b"plonky2";

        let padded_messages = (0..num_rounds).map(|_| SHA512Util::pad(msg));
        let mut expected_w = Vec::new();

        for (message, register) in padded_messages.zip_eq(data.public.padded_chunks.iter()) {
            let padded_msg = message
                .chunks_exact(8)
                .map(|slice| u64::from_be_bytes(slice.try_into().unwrap()))
                .map(u64_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            let pre_processed = SHA512Util::pre_process(&message);

            expected_w.push(pre_processed);
        }

        for end_bit in data.public.end_bits.iter() {
            writer.write(&end_bit, &GoldilocksField::ONE, 0);
        }

        writer.write_global_instructions(&stark.air_data);
        for i in 0..num_rows {
            writer.write_row_instructions(&stark.air_data, i);
        }

        // Compare the expected values with the ones in the trace.
        for (r, exp_w) in expected_w.iter().enumerate() {
            for (j, exp) in exp_w.iter().enumerate() {
                let w_i_value = u64::from_le_bytes(
                    writer
                        .read(&w_i, 80 * r + j)
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
    pub struct SHA512Test;

    impl AirParameters for SHA512Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 600;
        const EXTENDED_COLUMNS: usize = 342;
    }

    #[test]
    fn test_sha512_byte_stark() {
        type L = SHA512Test;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha512_processing", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let num_rounds = 1 << 2;
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<U64Register>(16))
            .collect::<Vec<_>>();
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let hash_state = builder.sha512(&padded_chunks, &end_bits);

        let num_rows = (80 * num_rounds).ilog2() as usize + 1;
        let stark = builder.build::<C, 2>(num_rows);

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let rec_data = recursive_builder.build::<Config>();

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let msg = b"plonky2";
        // let expected_digest = "7c6159dd615db8c15bc76e23d36106e77464759979a0fcd1366e531f552cfa0852dbf5c832f00bb279cbc945b44a132bff3ed0028259813b6a07b57326e88c87";

        let padded_messages = (0..num_rounds).map(|_| SHA512Util::pad(msg));

        for ((message, register), h_arr) in padded_messages
            .zip_eq(padded_chunks.iter())
            .zip(hash_state.iter())
        {
            let padded_msg = message
                .chunks_exact(8)
                .map(|slice| u64::from_be_bytes(slice.try_into().unwrap()))
                .map(u64_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            // let pre_processed = sha512Gadget::process_inputs(&message);
            // let state = sha512Gadget::compress_round(INITIAL_HASH, &pre_processed, ROUND_CONSTANTS)
            //     .map(u64_to_le_field_bytes);
            // writer.write_slice(h_arr, &state.concat(), 0);

            // expected_w.push(pre_processed);
        }
        for end_bit in end_bits.iter() {
            writer.write(&end_bit, &GoldilocksField::ONE, 0);
        }

        timed!(timing, "write input", {
            writer.write_global_instructions(&stark.air_data);
            (0..num_rounds).for_each(|r| {
                for k in 0..80 {
                    let i = r * 80 + k;
                    writer.write_row_instructions(&stark.air_data, i);
                }
            });
        });

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
