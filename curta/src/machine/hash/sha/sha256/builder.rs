use super::data::{SHA256Data, SHA256Memory, SHA256PublicData, SHA256TraceData};
use super::{INITIAL_HASH, ROUND_CONSTANTS};
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U32Register;
use crate::chip::uint::util::u32_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::math::prelude::*;

impl<L: AirParameters> BytesBuilder<L>
where
    L::Instruction: UintInstructions,
{
    pub fn sha_256_data(&mut self, num_rounds: usize) -> SHA256Data {
        // Convert the number of rounds to a field element.
        let num_round_element = self.constant(&L::Field::from_canonical_usize(num_rounds));

        // Initialize the initial hash and set it to the constant value.
        let initial_hash =
            self.constant_array::<U32Register>(&INITIAL_HASH.map(u32_to_le_field_bytes));

        // Initialize the round constants and set them to the constant value.
        let round_constant_values =
            self.constant_array::<U32Register>(&ROUND_CONSTANTS.map(u32_to_le_field_bytes));

        // Store the round constants in a slice to be able to load them in the trace.
        let round_constants = self.initialize_slice(
            &round_constant_values,
            &Time::zero(),
            Some(num_round_element),
        );

        // Initialize shift read multiplicities with zeros.
        let mut shift_read_mult = [L::Field::ZERO; 64];
        // Add multiplicities for reading the elements w[i-15].
        for mult in shift_read_mult.iter_mut().skip(1).take(48) {
            *mult += L::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-2].
        for mult in shift_read_mult.iter_mut().skip(14).take(48) {
            *mult += L::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-16].
        for mult in shift_read_mult.iter_mut().take(48) {
            *mult += L::Field::ONE;
        }
        // Add multiplicities for reading the elements w[i-7].
        for mult in shift_read_mult.iter_mut().skip(16 - 7).take(48) {
            *mult += L::Field::ONE;
        }

        let shift_read_values = self.constant_array::<ElementRegister>(&shift_read_mult);

        let shift_read_mult =
            self.initialize_slice(&shift_read_values, &Time::zero(), Some(num_round_element));

        let w = self.uninit_slice();
        let padded_chunks = (0..num_rounds)
            .map(|_| self.alloc_array_public::<U32Register>(16))
            .collect::<Vec<_>>();
        let dummy_entry = self.constant::<U32Register>(&[L::Field::ZERO; 4]);

        let dummy_index = self.constant(&L::Field::from_canonical_u8(64));

        let num_dummy_reads =
            self.constant::<ElementRegister>(&L::Field::from_canonical_usize(16 * 4 + 64 - 16));

        for (i, padded_chunk) in padded_chunks.iter().enumerate() {
            for (j, word) in padded_chunk.iter().enumerate().take(16) {
                self.store(&w.get(j), word, &Time::constant(i), None);
            }
            self.store(
                &w.get(64),
                dummy_entry,
                &Time::constant(i),
                Some(num_dummy_reads),
            );
        }

        // Initialize cycles to generate process id and is_processing flag.
        let cycle_16 = self.cycle(4);
        let cycle_64 = self.cycle(6);

        // `process_id` is a register is computed by counting the number of cycles. We do this by
        // setting `process_id` to be the cumulatibe sum of the `end_bit` of each cycle.
        let process_id = self.alloc::<ElementRegister>();
        self.set_to_expression_transition(
            &process_id.next(),
            process_id.expr() + cycle_64.end_bit.expr(),
        );
        // The array index register can be computed as `clock - process_id * 64`.
        let index =
            self.expression(self.clk.expr() - process_id.expr() * L::Field::from_canonical_u32(64));

        // Preprocessing happens in steps 16..64 of each 64-cycle. We compute this register by
        // having an accumnumator so that:
        //    - `is_preprocessing` becomes `0` at the beginning of each 64 cycle.
        //    - `is_preprocessing` becomes `1` at the end of every 16 cycle unless this coincides
        //       with the end of a 64-cycle.
        //    - otherwise, `is_preprocessing` remains the same.
        let is_preprocessing = self.alloc::<BitRegister>();
        self.set_to_expression_transition(
            &is_preprocessing.next(),
            cycle_64.end_bit.not_expr()
                * (cycle_16.end_bit.expr() + cycle_16.end_bit.not_expr() * is_preprocessing.expr()),
        );

        // Allocate end_bits for public input.
        let end_bits_input = self.alloc_array_public::<BitRegister>(num_rounds);
        let reg_64 = self.constant(&L::Field::from_canonical_u8(64));
        let end_bit = self.uninit_slice();
        for (i, end_bit_val) in end_bits_input.iter().enumerate() {
            self.store(&end_bit.get(i), end_bit_val, &Time::zero(), Some(reg_64));
        }

        let public = SHA256PublicData {
            initial_hash,
            padded_chunks,
            end_bits: end_bits_input,
        };

        let trace = SHA256TraceData {
            is_preprocessing,
            process_id,
            cycle_64_end_bit: cycle_64.end_bit,
            index,
        };

        let memory = SHA256Memory {
            round_constants,
            w,
            shift_read_mult,
            end_bit,
            dummy_index,
        };
        SHA256Data {
            public,
            trace,
            memory,
            num_chunks: num_rounds,
        }
    }

    pub fn sha_256_preprocessing(&mut self, data: &SHA256Data) -> U32Register {
        let w = &data.memory.w;
        let index = data.trace.index;
        let dummy_index = data.memory.dummy_index;
        let process_id = data.trace.process_id;
        let is_preprocessing = data.trace.is_preprocessing;
        let shift_read_mult = &data.memory.shift_read_mult;

        let time = Time::from_element(process_id);

        let shifted_index = |i: u32, builder: &mut BytesBuilder<L>| {
            builder.expression(
                is_preprocessing.expr() * (index.expr() - L::Field::from_canonical_u32(i))
                    + is_preprocessing.not_expr() * dummy_index.expr(),
            )
        };

        // Calculate the value:
        // s_0 = w_i_minus_15.rotate_right(7) ^ w_i_minus_15.rotate_right(18) ^ (w_i_minus_15 >> 3)
        let i_m_15 = shifted_index(15, self);
        let w_i_minus_15 = self.load(&w.get_at(i_m_15), &time);
        let w_i_minus_15_rotate_7 = self.rotate_right(w_i_minus_15, 7);
        let w_i_minus_15_rotate_18 = self.rotate_right(w_i_minus_15, 18);
        let w_i_minus_15_shr_3 = self.shr(w_i_minus_15, 3);

        let mut s_0 = self.xor(&w_i_minus_15_rotate_7, &w_i_minus_15_rotate_18);
        s_0 = self.xor(&s_0, &w_i_minus_15_shr_3);

        // Calculate the value:
        // s_1 = w_i_minus_2.rotate_right(17) ^ w_i_minus_2.rotate_right(19) ^ (w_i_minus_2 >> 10)
        let i_m_2 = shifted_index(2, self);
        let w_i_minus_2 = self.load(&w.get_at(i_m_2), &time);
        let w_i_minus_2_rotate_17 = self.rotate_right(w_i_minus_2, 17);
        let w_i_minus_2_rotate_19 = self.rotate_right(w_i_minus_2, 19);
        let w_i_minus_2_shr_10 = self.shr(w_i_minus_2, 10);

        let mut s_1 = self.xor(&w_i_minus_2_rotate_17, &w_i_minus_2_rotate_19);
        s_1 = self.xor(&s_1, &w_i_minus_2_shr_10);

        // Calculate the value:
        // w_i = w_i_minus_16 + s_0 + w_i_minus_7 + s_1
        let i_m_16 = shifted_index(16, self);
        let w_i_mimus_16 = self.load(&w.get_at(i_m_16), &time);
        let i_m_7 = shifted_index(7, self);
        let w_i_mimus_7 = self.load(&w.get_at(i_m_7), &time);
        let mut w_i_pre_process = self.add(w_i_mimus_16, s_0);
        w_i_pre_process = self.add(w_i_pre_process, w_i_mimus_7);
        w_i_pre_process = self.add(w_i_pre_process, s_1);

        let i_idx = self.select(is_preprocessing, &dummy_index, &index);
        let w_i_read = self.load(&w.get_at(i_idx), &time);

        let w_i = self.select(is_preprocessing, &w_i_pre_process, &w_i_read);

        let reading_mult = self.load(&shift_read_mult.get_at(index), &Time::zero());
        self.store(&w.get_at(index), w_i, &time, Some(reading_mult));

        w_i
    }

    /// The processing step of a SHA256 round.
    pub fn sha_processing(
        &mut self,
        w_i: U32Register,
        data: &SHA256Data,
    ) -> Vec<ArrayRegister<U32Register>> {
        let hash_state_public = (0..data.num_chunks)
            .map(|_| self.alloc_array_public::<U32Register>(8))
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
        let cycle_end_bit = data.trace.cycle_64_end_bit;

        let round_constant = self.load(&round_constant.get_at(index), &Time::zero());

        // Initialize working variables
        let state = self.alloc_array::<U32Register>(8);
        for (h, h_init) in state.iter().zip(initial_hash.iter()) {
            self.set_to_expression_first_row(&h, h_init.expr());
        }
        // Initialize working variables and set them to the inital hash in the first row.
        let vars = self.alloc_array::<U32Register>(8);
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

        // Calculate sum_1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25).
        let e_rotate_6 = self.rotate_right(e, 6);
        let e_rotate_11 = self.rotate_right(e, 11);
        let e_rotate_25 = self.rotate_right(e, 25);
        let mut sum_1 = self.xor(e_rotate_6, e_rotate_11);
        sum_1 = self.xor(sum_1, e_rotate_25);

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

        // Calculate sum_0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22).
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

        let state_plus_vars = state
            .iter()
            .zip(vars_next.iter())
            .map(|(s, v)| self.add(s, *v))
            .collect::<Vec<_>>();

        // Store the new state values
        let flag = Some(cycle_end_bit.as_element());
        let process_id = data.trace.process_id;
        for (i, state_plus_var) in state_plus_vars.iter().enumerate() {
            self.store(
                &state_ptr.get(i),
                *state_plus_var,
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
    use crate::chip::hash::sha::sha256::SHA256Gadget;
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
    use crate::chip::uint::operations::instruction::UintInstruction;
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
        let data = builder.sha_256_data(num_rounds);

        let w_i = builder.sha_256_preprocessing(&data);

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
        // let expected_digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

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
        const EXTENDED_COLUMNS: usize = 360;
    }

    #[test]
    fn test_sha256_processing() {
        type L = SHA256Test;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha256_processing", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let num_rounds = 1 << 13;
        let data = builder.sha_256_data(num_rounds);

        let w_i = builder.sha_256_preprocessing(&data);
        let hash_state = builder.sha_processing(w_i, &data);

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
        // let expected_digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let padded_messages = (0..num_rounds).map(|_| SHA256Gadget::pad(msg));
        let mut expected_w = Vec::new();

        for ((message, register), h_arr) in padded_messages
            .zip_eq(data.public.padded_chunks.iter())
            .zip(hash_state.iter())
        {
            let padded_msg = message
                .chunks_exact(4)
                .map(|slice| u32::from_be_bytes(slice.try_into().unwrap()))
                .map(u32_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            let pre_processed = SHA256Gadget::process_inputs(&message);
            let state = SHA256Gadget::compress_round(INITIAL_HASH, &pre_processed, ROUND_CONSTANTS)
                .map(u32_to_le_field_bytes);
            writer.write_array(h_arr, state, 0);

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

        timed!(
            timing,
            "insert inputs",
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
        );

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
