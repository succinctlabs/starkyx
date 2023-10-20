use super::data::SHA256Data;
use super::{INITIAL_HASH, ROUND_CONSTANTS};
use crate::chip::memory::time::Time;
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
        let state = self.uninit_slice();

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
        let padded_messages = (0..num_rounds)
            .map(|_| self.alloc_array_public::<U32Register>(16))
            .collect::<Vec<_>>();
        let dummy_entry = self.constant::<U32Register>(&[L::Field::ZERO; 4]);

        let dummy_index = self.constant(&L::Field::from_canonical_u8(64));

        let num_dummy_reads =
            self.constant::<ElementRegister>(&L::Field::from_canonical_usize(16 * 4 + 64 - 16));

        for (i, padded_message) in padded_messages.iter().enumerate() {
            for (j, word) in padded_message.iter().enumerate().take(16) {
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

        SHA256Data {
            state,
            initial_hash,
            round_constants,
            w,
            index,
            is_preprocessing,
            padded_messages,
            process_id,
            dummy_index,
            shift_read_mult,
        }
    }

    pub fn sha_256_preprocessing(&mut self, data: &SHA256Data) -> U32Register {
        let w = &data.w;
        let index = data.index;
        let dummy_index = data.dummy_index;
        let process_id = data.process_id;
        let is_preprocessing = data.is_preprocessing;
        let shift_read_mult = &data.shift_read_mult;

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
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
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

        const NUM_FREE_COLUMNS: usize = 200;
        const EXTENDED_COLUMNS: usize = 117;
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

        // Read the round constant to make the bus argument work.
        let _ = builder.load(&data.round_constants.get_at(data.index), &Time::zero());

        let num_rows = 64 * num_rounds;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let msg = b"abc";
        // let expected_digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let padded_messages = (0..num_rounds).map(|_| SHA256Gadget::pad(msg));
        let mut expected_w = Vec::new();

        for (message, register) in padded_messages.zip_eq(data.padded_messages.iter()) {
            let padded_msg = message
                .chunks_exact(4)
                .map(|slice| u32::from_be_bytes(slice.try_into().unwrap()))
                .map(u32_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            let pre_processed = SHA256Gadget::process_inputs(&message);

            expected_w.push(pre_processed);
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
}
