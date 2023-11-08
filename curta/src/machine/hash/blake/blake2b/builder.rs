use super::data::{BLAKE2BData, SHA256Data, SHA256Memory, SHA256PublicData, SHA256TraceData};
use super::register::SHA256DigestRegister;
use super::{INITIAL_HASH, ROUND_CONSTANTS};
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::{ByteArrayRegister, U32Register, U64Register};
use crate::chip::uint::util::{u32_to_le_field_bytes, u64_to_le_field_bytes};
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::machine::hash::blake::blake2b::data::BLAKE2BTraceData;
use crate::machine::hash::blake::blake2b::{
    NUM_PERMUTATIONS, PERMUTATION_SIZE, SIGMA_PERMUTATIONS,
};
use crate::math::prelude::*;

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

impl<L: AirParameters> BytesBuilder<L>
where
    L::Instruction: UintInstructions,
{
    fn cycles_end_bits(builder: &mut BytesBuilder<L>) -> (BitRegister, BitRegister) {
        let cycle_8 = builder.cycle(3);
        let loop_3 = builder.api.loop_instr(3);

        let cycle_96_end_bit = {
            let cycle_32 = builder.cycle(5);
            builder.mul(loop_3.get_iteration_reg(2), cycle_32.end_bit)
        };

        (cycle_8.end_bit, cycle_96_end_bit)
    }

    pub fn blake2b(
        &mut self,
        padded_chunks: &[ArrayRegister<U32Register>],
        msg_lens: &[ArrayRegister<U64Register>],
        end_bits: &ArrayRegister<BitRegister>,
        initial_hash: &ArrayRegister<ByteArrayRegister<8>>,
        initial_hash_compress: &ArrayRegister<ByteArrayRegister<8>>,
    ) -> Vec<SHA256DigestRegister> {
        let data = self.blake2b_data(
            padded_chunks,
            msg_lens,
            end_bits,
            initial_hash,
            initial_hash_compress,
        );

        let h = self.alloc_array::<U64Register>(8);
        for (h_i, initial_hash_i) in h.iter().zip(data.public.initial_hash.iter()) {
            self.set_to_expression_first_row(&h_i, initial_hash_i.expr());
        }

        self.blake2b_compress(h, &data)
    }

    pub fn blake2b_data(
        &mut self,
        padded_chunks: &[ArrayRegister<U64Register>],
        msg_lens: &[ArrayRegister<U64Register>],
        end_bits: &ArrayRegister<BitRegister>,
        initial_hash: &ArrayRegister<ByteArrayRegister<8>>,
        initial_hash_compress: &ArrayRegister<ByteArrayRegister<8>>,
    ) -> SHA256Data {
        assert_eq!(padded_chunks.len(), end_bits.len());
        let num_rounds = padded_chunks.len();
        // Convert the number of rounds to a field element.
        let num_round_element = self.constant(&L::Field::from_canonical_usize(num_rounds));

        let double_num_round_element = self.add(num_round_element, num_round_element);

        // Initialize the initial hash and set it to the constant value.
        let initial_hash =
            self.constant_array::<ByteArrayRegister<8>>(&INITIAL_HASH.map(u64_to_le_field_bytes));

        // Initialize the initial hash compress and set it to the constant value.
        let initial_hash_compress =
            self.constant_array::<ByteArrayRegister<8>>(&INITIAL_HASH.map(u64_to_le_field_bytes));

        // Initialize the permutation constants memory
        let mut permutation_values = Vec::new();
        for permuation_const in SIGMA_PERMUTATIONS {
            permutation_values.push(self.constant_array::<ElementRegister>(
                &permuation_const.map(L::Field::from_canonical_u8),
            ));
        }

        // Store the round constants in a slice to be able to load them in the trace.
        // The first two sigma permutations will be accessed twice per round.
        // All the othes will be accessed only once.
        let mut permutations = Vec::new();
        for (i, permuation_value) in permutation_values.iter().enumerate() {
            permutations.push(self.initialize_slice(
                permuation_value,
                &Time::zero(),
                Some(num_round_element),
            ));
        }

        // Initialize the m memory
        let m = self.uninit_slice();

        for (i, padded_chunk) in padded_chunks.iter().enumerate() {
            for (j, word) in padded_chunk.iter().enumerate().take(32) {
                self.store(&m.get(j), word, &Time::constant(i), None);
            }
        }

        let (cycle_8_end_bit, cycle_96_end_bit) = Self::cycles_end_bits(self);

        // `process_id` is a register is computed by counting the number of cycles. We do this by
        // setting `process_id` to be the cumulative sum of the `end_bit` of each cycle.
        let process_id: ElementRegister = self.alloc::<ElementRegister>();
        self.set_to_expression_first_row(&process_id, L::Field::ZERO.into());
        self.set_to_expression_transition(
            &process_id.next(),
            process_id.expr() + cycle_96_end_bit.expr(),
        );

        let mix_iteration: ElementRegister = self.alloc::<ElementRegister>();
        self.set_to_expression_first_row(&mix_iteration, L::Field::ZERO.into());
        self.set_to_expression_transition(
            &mix_iteration.next(),
            (cycle_96_end_bit.expr() * L::Field::ZERO.into())
                + cycle_96_end_bit.not_expr()
                    * (cycle_8_end_bit.not_expr() * mix_iteration.expr()
                        + cycle_8_end_bit.expr() * (mix_iteration.expr() + L::Field::ONE.into())),
        );

        let mix_index = self.alloc::<ElementRegister>();
        self.set_to_expression_first_row(&mix_index, L::Field::ZERO.into());
        self.set_to_expression_transition(
            &mix_index.next(),
            cycle_8_end_bit.not_expr() * (mix_index.expr() + L::Field::ONE.into())
                + cycle_8_end_bit.expr() * L::Field::ZERO.into(),
        );

        // The array index register can be computed as `clock - process_id * CYCLE_LENGTH`.
        let clk = Self::clk(self);
        let index =
            self.expression(clk.expr() - process_id.expr() * L::Field::from_canonical_usize(96));

        let is_compress_initialize = self.alloc::<BitRegister>();
        self.set_to_expression_first_row(&is_compress_initialize, L::Field::ONE.into());
        self.set_to_expression_transition(&is_compress_initialize.next(), cycle_96_end_bit.expr());

        // Allocate end_bits for public input.
        let reg_96 = self.constant(&L::Field::from_canonical_u8(96));
        let end_bit = self.uninit_slice();
        for (i, end_bit_val) in end_bits.iter().enumerate() {
            self.store(&end_bit.get(i), end_bit_val, &Time::zero(), Some(reg_96));
        }

        let public = Blake2bPublicData {
            initial_hash,
            initial_hash_compress,
            padded_chunks: padded_chunks.to_vec(),
            end_bits: *end_bits,
        };

        let trace = BLAKE2BTraceData {
            is_compress_initialize,
            process_id,
            cycle_8_end_bit,
            cycle_96_end_bit,
            mix_iteration,
            mix_index,
        };

        let memory = Blake2bMemory {
            permutations,
            m,
            end_bit,
        };
        Blake2bData {
            public,
            trace,
            memory,
            num_chunks: num_rounds,
        }
    }

    pub fn blake2b_compress_initialize(
        &mut self,
        v: ArrayRegister<U64Register>,
        h: ArrayRegister<U64Register>,
        data: &BLAKE2BData,
    ) {
        let is_compress_initialize = data.trace.is_compress_initialize;

        for i in 0..16 {
            let v_i_read = self.load(&data.memory.v.get_at(i), &Time::zero());
            let v_i_save = if i < 8 {
                self.select(is_compress_initialize, h.get_at(i), v_i_read)
            } else {
                self.select(
                    is_compress_initialize,
                    data.public.compress_initial_hash.get_at(i - 8),
                    v_i_read,
                )
            };

            self.store(data.memory.v.get_at(i), v_i_save, &Time::zero());
        }
    }

    /// The processing step of a SHA256 round.
    pub fn blake2b_processing(
        &mut self,
        h: &ArrayRegister<U64Register>,
        data: &BLAKE2BData,
    ) -> Vec<SHA256DigestRegister> {
        let v = self.alloc_array::<U64Register>(16);

        for (v_i, h_i) in v.iter().take(8).zip(h.iter()) {
            self.set_to_expression_first_row(&v_i, h_i.expr());
        }

        for (v_i, public_input_i) in v
            .iter()
            .skip(8)
            .zip(data.public.initial_hash_compress.iter())
        {
            self.set_to_expression_first_row(&v_i, public_input_i.expr());
        }

        let permutation = self.load(
            &data.memory.permutations.get_at(data.trace.mix_iteration),
            &Time::zero(),
        );
        let state_indices = self.load(
            &data.memory.state_indices.get_at(data.trace.mix_index),
            &Time::zero(),
        );
        let index_1 = state_indices.get(0);
        let index_2 = state_indices.get(1);
        let index_3 = state_indices.get(2);
        let index_4 = state_indices.get(3);

        let state_1 = v.from_element(index_1);

        let state_1 = self.load(data.memory.v.get_at(index_1), &Time::zero());
        let state_2 = self.load(data.memory.v.get_at(index_2), &Time::zero());
        let state_3 = self.load(data.memory.v.get_at(index_3), &Time::zero());
        let state_4 = self.load(data.memory.v.get_at(index_4), &Time::zero());

        let m_index_1 = self.mul(data.trace.mix_index, L::Field::from_canonical_usize(2));
        let m_index_2 = self.add(m_index_1, L::Field::ONE.into());

        let m_1 = self.load(data.memory.m.get_at(m_index_1), &Time::zero());
        let m_2 = self.load(data.memory.m.get_at(m_index_2), &Time::zero());

        self.blake2b_mix(state_1, state_2, state_3, state_4, m_1, m_2);
    }

    pub fn blake2b_mix(
        &mut self,
        v_a: &mut U64Register,
        v_b: &mut U64Register,
        v_c: &mut U64Register,
        v_d: &mut U64Register,
        x: &U64Register,
        y: &U64Register,
    ) {
        *v_a = self.add(*v_a, *v_b);
        *v_a = self.add(*v_a, *x);

        *v_d = self.xor(*v_d, *v_a);
        *v_d = self.rotate_right(*v_d, 32);

        *v_c = self.add(*v_c, *v_d);

        *v_b = self.xor(*v_b, *v_c);
        *v_b = self.rotate_right(*v_b, 24);

        *v_a = self.add(*v_a, *v_b);
        *v_a = self.add(*v_a, *y);

        *v_d = self.xor(*v_d, *v_a);
        *v_d = self.rotate_right(*v_d, 16);

        *v_c = self.add(*v_c, *v_d);

        *v_b = self.xor(*v_b, *v_c);
        *v_b = self.rotate_right(*v_b, 63);
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
            let state = SHA256Gadget::compress_round(INITIAL_HASH, &pre_processed, ROUND_CONSTANTS)
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
