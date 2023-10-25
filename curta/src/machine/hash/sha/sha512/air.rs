use super::register::SHA512DigestRegister;
use super::SHA512;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::{u64_from_le_field_bytes, u64_to_le_field_bytes};
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::machine::hash::sha::algorithm::SHAir;

impl<L: AirParameters> SHAir<BytesBuilder<L>, 80> for SHA512
where
    L::Instruction: UintInstructions,
{
    type Value = <U64Register as Register>::Value<L::Field>;
    type Variable = U64Register;
    type StateVariable = SHA512DigestRegister;
    type StatePointer = Slice<U64Register>;

    fn int_to_field_value(int: Self::Integer) -> Self::Value {
        u64_to_le_field_bytes(int)
    }

    fn field_value_to_int(value: &Self::Value) -> Self::Integer {
        u64_from_le_field_bytes(value)
    }

    fn clk(builder: &mut BytesBuilder<L>) -> ElementRegister {
        builder.clk
    }

    fn cycles_end_bits(builder: &mut BytesBuilder<L>) -> (BitRegister, BitRegister) {
        let cycle_16 = builder.cycle(4);
        let cycle_80_end_bit = {
            let loop_5 = builder.api.loop_instr(5);
            let five_end_bit = loop_5.get_iteration_reg(4);
            builder.mul(five_end_bit, cycle_16.end_bit)
        };

        (cycle_16.end_bit, cycle_80_end_bit)
    }

    fn load_state(
        builder: &mut BytesBuilder<L>,
        hash_state_public: &[Self::StateVariable],
    ) -> Self::StatePointer {
        let state_ptr = builder.uninit_slice();

        for (i, h_slice) in hash_state_public.iter().enumerate() {
            for (j, h) in h_slice.iter().enumerate() {
                builder.free(&state_ptr.get(j), h, &Time::constant(i));
            }
        }

        state_ptr
    }

    fn store_state(
        builder: &mut BytesBuilder<L>,
        state_ptr: &Self::StatePointer,
        state_next: [Self::Variable; 8],
        time: &Time<L::Field>,
        flag: Option<ElementRegister>,
    ) {
        for (i, element) in state_next.iter().enumerate() {
            builder.store(&state_ptr.get(i), *element, time, flag);
        }
    }

    fn preprocessing_step(
        builder: &mut BytesBuilder<L>,
        w_i_minus_15: Self::Variable,
        w_i_minus_2: Self::Variable,
        w_i_mimus_16: Self::Variable,
        w_i_mimus_7: Self::Variable,
    ) -> Self::Variable {
        // Calculate the value:
        // s_0 = w_i_minus_15.rotate_right(1) ^ w_i_minus_15.rotate_right(8) ^ (w_i_minus_15 >> 7)
        let w_i_minus_15_rotate_1 = builder.rotate_right(w_i_minus_15, 1);
        let w_i_minus_15_rotate_8 = builder.rotate_right(w_i_minus_15, 8);
        let w_i_minus_15_shr_7 = builder.shr(w_i_minus_15, 7);

        let mut s_0 = builder.xor(&w_i_minus_15_rotate_1, &w_i_minus_15_rotate_8);
        s_0 = builder.xor(&s_0, &w_i_minus_15_shr_7);

        let w_i_minus_2_rotate_19 = builder.rotate_right(w_i_minus_2, 19);
        let w_i_minus_2_rotate_61 = builder.rotate_right(w_i_minus_2, 61);
        let w_i_minus_2_shr_6 = builder.shr(w_i_minus_2, 6);

        let mut s_1 = builder.xor(&w_i_minus_2_rotate_19, &w_i_minus_2_rotate_61);
        s_1 = builder.xor(&s_1, &w_i_minus_2_shr_6);

        // Calculate the value:
        // w_i = w_i_minus_16 + s_0 + w_i_minus_7 + s_1
        let mut w_i_pre_process = builder.add(w_i_mimus_16, s_0);
        w_i_pre_process = builder.add(w_i_pre_process, w_i_mimus_7);
        builder.add(w_i_pre_process, s_1)
    }

    fn processing_step(
        builder: &mut BytesBuilder<L>,
        vars: ArrayRegister<Self::Variable>,
        w_i: Self::Variable,
        round_constant: Self::Variable,
    ) -> Vec<Self::Variable> {
        let a = vars.get(0);
        let b = vars.get(1);
        let c = vars.get(2);
        let d = vars.get(3);
        let e = vars.get(4);
        let f = vars.get(5);
        let g = vars.get(6);
        let h = vars.get(7);

        // Calculate S1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41).
        let e_rotate_14 = builder.rotate_right(e, 14);
        let e_rotate_18 = builder.rotate_right(e, 18);
        let e_rotate_41 = builder.rotate_right(e, 41);
        let mut sum_1 = builder.xor(e_rotate_14, e_rotate_18);
        sum_1 = builder.xor(sum_1, e_rotate_41);

        // Calculate ch = (e & f) ^ (!e & g).
        let e_and_f = builder.and(&e, &f);
        let not_e = builder.not(e);
        let not_e_and_g = builder.and(&not_e, &g);
        let ch = builder.xor(&e_and_f, &not_e_and_g);

        // Calculate temp_1 = h + sum_1 + ch + round_constant + w.
        let mut temp_1 = builder.add(h, sum_1);
        temp_1 = builder.add(temp_1, ch);
        temp_1 = builder.add(temp_1, round_constant);
        temp_1 = builder.add(temp_1, w_i);

        // Calculate S0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39).
        let a_rotate_28 = builder.rotate_right(a, 28);
        let a_rotate_34 = builder.rotate_right(a, 34);
        let a_rotate_39 = builder.rotate_right(a, 39);
        let mut sum_0 = builder.xor(a_rotate_28, a_rotate_34);
        sum_0 = builder.xor(sum_0, a_rotate_39);

        // Calculate maj = (a & b) ^ (a & c) ^ (b & c);
        let a_and_b = builder.and(a, b);
        let a_and_c = builder.and(a, c);
        let b_and_c = builder.and(b, c);
        let mut maj = builder.xor(a_and_b, a_and_c);
        maj = builder.xor(maj, b_and_c);

        // Calculate temp_2 = sum_0 + maj.
        let temp_2 = builder.add(sum_0, maj);

        // Calculate the next cycle values.
        let a_next = builder.add(temp_1, temp_2);
        let b_next = a;
        let c_next = b;
        let d_next = c;
        let e_next = builder.add(d, temp_1);
        let f_next = e;
        let g_next = f;
        let h_next = g;

        vec![
            a_next, b_next, c_next, d_next, e_next, f_next, g_next, h_next,
        ]
    }

    fn absorb(
        builder: &mut BytesBuilder<L>,
        state: ArrayRegister<Self::Variable>,
        vars_next: &[Self::Variable],
    ) -> [Self::Variable; 8] {
        state
            .iter()
            .zip(vars_next.iter())
            .map(|(s, v)| builder.add(s, *v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use core::iter;

    use itertools::Itertools;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use serde::{Deserialize, Serialize};

    use crate::chip::memory::time::Time;
    use crate::chip::register::bit::BitRegister;
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::register::U64Register;
    use crate::chip::uint::util::{u64_from_le_field_bytes, u64_to_le_field_bytes};
    use crate::chip::AirParameters;
    use crate::machine::builder::Builder;
    use crate::machine::bytes::builder::BytesBuilder;
    use crate::machine::hash::sha::algorithm::{SHAPure, SHAir};
    use crate::machine::hash::sha::builder::SHABuilder;
    use crate::machine::hash::sha::sha512::{INITIAL_HASH, ROUND_CONSTANTS, SHA512};
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::math::prelude::*;
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
        let data = SHA512::data(&mut builder, &padded_chunks, &end_bits);

        let w_i = SHA512::preprocessing(&mut builder, &data);

        // Dummy reads and writes to make the bus argument work.
        builder.load(
            &data.memory.round_constants.get_at(data.trace.index),
            &Time::zero(),
        );
        builder.load(
            &data.memory.end_bit.get_at(data.trace.process_id),
            &Time::zero(),
        );

        let num_rows = data.degree;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let msg_1 = b"plonky2";
        let msg_2 = b"";

        let padded_messages = [SHA512::pad(msg_1), SHA512::pad(msg_2)];
        let mut expected_w = Vec::new();

        for (message, register) in padded_messages
            .iter()
            .zip_eq(data.public.padded_chunks.iter())
        {
            let padded_msg = message
                .chunks_exact(8)
                .map(|slice| u64::from_be_bytes(slice.try_into().unwrap()))
                .map(u64_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            let pre_processed = SHA512::pre_process(message);
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
                assert_eq!(w_i_value, *exp, "w at row {} is incorrect", 80 * r + j);
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

        const NUM_FREE_COLUMNS: usize = 1190;
        const EXTENDED_COLUMNS: usize = 651;
    }

    fn test_sha512<'a, I: IntoIterator<Item = &'a [u8]>, J: IntoIterator<Item = &'a str>>(
        messages: I,
        expected_digests: J,
    ) {
        type L = SHA512Test;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let mut end_bits_values = Vec::new();
        let padded_chunks_values = messages
            .into_iter()
            .flat_map(|msg| {
                let padded_msg = SHA512::pad(msg);
                let num_chunks = padded_msg.len() / 128;
                end_bits_values.extend_from_slice(&vec![GoldilocksField::ZERO; num_chunks - 1]);
                end_bits_values.push(GoldilocksField::ONE);
                padded_msg
            })
            .collect::<Vec<_>>();

        assert_eq!(end_bits_values.len() * 128, padded_chunks_values.len());
        let num_rounds = end_bits_values.len();
        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha512", log::Level::Debug);

        // Build the stark.
        let mut builder = BytesBuilder::<L>::new();
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<U64Register>(16))
            .collect::<Vec<_>>();
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let hash_state = builder.sha::<SHA512, 80>(&padded_chunks, &end_bits);

        let num_rows_degree = (80 * num_rounds).ilog2() as usize + 1;
        let num_rows = 1 << num_rows_degree;
        let stark = builder.build::<C, 2>(num_rows);

        // Build the recursive circuit.
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let rec_data = recursive_builder.build::<Config>();

        // Write trace.
        let writer = TraceWriter::new(&stark.air_data, num_rows);

        timed!(timing, "write input", {
            writer.write_global_instructions(&stark.air_data);
            for i in 0..num_rows {
                writer.write_row_instructions(&stark.air_data, i);
            }
        });

        let mut current_state = INITIAL_HASH;
        for ((((message, register), h_arr), end_bit), end_bit_value) in padded_chunks_values
            .chunks_exact(128)
            .zip_eq(padded_chunks.iter())
            .zip(hash_state.iter())
            .zip_eq(end_bits.iter())
            .zip_eq(end_bits_values.iter())
        {
            let padded_msg = message
                .chunks_exact(8)
                .map(|slice| u64::from_be_bytes(slice.try_into().unwrap()))
                .map(u64_to_le_field_bytes::<GoldilocksField>)
                .collect::<Vec<_>>();

            writer.write_array(register, padded_msg, 0);

            let pre_processed = SHA512::pre_process(message);
            current_state = SHA512::process(current_state, &pre_processed, ROUND_CONSTANTS);
            let state = current_state.map(u64_to_le_field_bytes);
            if *end_bit_value == GoldilocksField::ONE {
                current_state = INITIAL_HASH;
            }
            writer.write_slice(h_arr, &state.concat(), 0);
            writer.write(&end_bit, end_bit_value, 0);
        }

        timed!(timing, "write input", {
            writer.write_global_instructions(&stark.air_data);
            for i in 0..num_rows {
                writer.write_row_instructions(&stark.air_data, i);
            }
        });

        // Compare expected digests with the trace values.
        let mut expected_digests = expected_digests.into_iter();
        for (state, end_bit) in hash_state.iter().zip_eq(end_bits_values) {
            if end_bit == GoldilocksField::ZERO {
                continue;
            }
            let digest = writer
                .read_array::<_, 8>(&state.as_array(), 0)
                .map(|x| u64_from_le_field_bytes(&x));
            let expected_digest = SHA512::decode(expected_digests.next().unwrap());
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

    #[test]
    fn test_sha512_short_message() {
        let msg = b"plonky2";
        let expected_digest = "7c6159dd615db8c15bc76e23d36106e77464759979a0fcd1366e531f552cfa0852dbf5c832f00bb279cbc945b44a132bff3ed0028259813b6a07b57326e88c87";
        let num_messages = 2;
        test_sha512(
            iter::repeat(msg).take(num_messages).map(|x| x.as_slice()),
            iter::repeat(expected_digest).take(num_messages),
        )
    }

    #[test]
    fn test_sha512_long_nessage() {
        let num_messages = 2;
        let msg = hex::decode("35c323757c20640a294345c89c0bfcebe3d554fdb0c7b7a0bdb72222c531b1ecf7ec1c43f4de9d49556de87b86b26a98942cb078486fdb44de38b80864c3973153756363696e6374204c616273").unwrap();
        let expected_digest = "4388243c4452274402673de881b2f942ff5730fd2c7d8ddb94c3e3d789fb3754380cba8faa40554d9506a0730a681e88ab348a04bc5c41d18926f140b59aed39";
        test_sha512(
            iter::repeat(msg.as_slice()).take(num_messages),
            iter::repeat(expected_digest).take(num_messages),
        )
    }

    #[test]
    fn test_sha512_changing_length_nessage() {
        let short_msg = b"plonky2";
        let short_expected_digest = "7c6159dd615db8c15bc76e23d36106e77464759979a0fcd1366e531f552cfa0852dbf5c832f00bb279cbc945b44a132bff3ed0028259813b6a07b57326e88c87";
        let long_msg = hex::decode("35c323757c20640a294345c89c0bfcebe3d554fdb0c7b7a0bdb72222c531b1ecf7ec1c43f4de9d49556de87b86b26a98942cb078486fdb44de38b80864c3973153756363696e6374204c616273").unwrap();
        let long_expected_digest = "4388243c4452274402673de881b2f942ff5730fd2c7d8ddb94c3e3d789fb3754380cba8faa40554d9506a0730a681e88ab348a04bc5c41d18926f140b59aed39";
        test_sha512(
            [
                short_msg.as_slice(),
                long_msg.as_slice(),
                short_msg.as_slice(),
            ],
            [
                short_expected_digest,
                long_expected_digest,
                short_expected_digest,
            ],
        );
    }
}
