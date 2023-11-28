use super::BLAKE2BAir;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::AirParameters;
use crate::machine::bytes::builder::BytesBuilder;

impl<L: AirParameters> BytesBuilder<L>
where
    L::Instruction: UintInstructions,
{
    fn blake2b(
        &mut self,
        padded_chunks: &[ArrayRegister<U64Register>],
        t_values: &ArrayRegister<U64Register>,
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: &ArrayRegister<ElementRegister>,
        num_messages: &ElementRegister,
    ) -> Vec<ArrayRegister<U64Register>> {
        BLAKE2BAir::blake2b(
            self,
            padded_chunks,
            t_values,
            end_bits,
            digest_bits,
            digest_indices,
            num_messages,
        )
    }
}

#[cfg(test)]
pub mod test_utils {

    use core::fmt::Debug;

    use itertools::Itertools;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::timed;
    use plonky2::util::log2_ceil;
    use plonky2::util::timing::TimingTree;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::util::u64_to_le_field_bytes;
    use crate::chip::AirParameters;
    use crate::machine::builder::Builder;
    use crate::machine::bytes::builder::BytesBuilder;
    use crate::machine::hash::blake::blake2b::pure::BLAKE2BPure;
    use crate::machine::hash::blake::blake2b::utils::BLAKE2BUtil;
    use crate::machine::hash::blake::blake2b::IV;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::math::prelude::*;
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};
    use crate::prelude::{AirWriter, AirWriterData};

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct BLAKE2BTest;

    impl AirParameters for BLAKE2BTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;
        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 1387;
        const EXTENDED_COLUMNS: usize = 834;
    }

    #[test]
    pub fn test_blake2b() {
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha", log::Level::Debug);

        const MAX_CHUNK_SIZE: u64 = 2;

        // Collect the public inputs
        let messages = vec![
            b"asfiwu4yrlisuhgluashdlowaualisdugylawi4thagasdf23uiraskdgbasjkdfhaliwhfrasdfaw4jhbskfjhsadkif325sgdsfawera".to_vec(),
            b"325623465236262asdagds326fdsfy3w456gery46462ialweurnawieyailughoiwabn4bkq23bh2jh5bkwaeublaieunrqi4awijbjkahtiqi3uwagastt3asgesgg3".to_vec(),
        ];

        let mut padded_chunks_values = Vec::new();
        let mut t_values_values = Vec::new();
        let mut end_bits_values = Vec::new();
        let mut digest_indices_values = Vec::new();

        let mut digest_index = 0;
        for message in messages.clone() {
            let msg_u64_limbs: Vec<[GoldilocksField; 8]> =
                BLAKE2BUtil::pad(&message, MAX_CHUNK_SIZE)
                    .chunks_exact(8)
                    .map(|x| {
                        x.iter()
                            .map(|y| GoldilocksField::from_canonical_u8(*y))
                            .collect_vec()
                            .try_into()
                            .unwrap()
                    })
                    .collect_vec();

            let msg_padded_chunks: Vec<[[GoldilocksField; 8]; 16]> = msg_u64_limbs
                .chunks_exact(16)
                .map(|x| x.try_into().unwrap())
                .collect_vec();

            let mut t_value = 0u64;
            let msg_len = message.len() as u64;
            for chunk in msg_padded_chunks.iter() {
                padded_chunks_values.push(*chunk);

                t_value += 128;

                let mut at_last_chunk = false;
                if t_value >= msg_len {
                    at_last_chunk = true;
                }

                t_values_values.push(if at_last_chunk {
                    u64_to_le_field_bytes(msg_len)
                } else {
                    u64_to_le_field_bytes(t_value)
                });

                end_bits_values.push(GoldilocksField::from_canonical_usize(
                    at_last_chunk as usize,
                ));
            }

            digest_index += msg_padded_chunks.len();
            digest_indices_values.push(GoldilocksField::from_canonical_usize(digest_index));
        }

        let num_messages_value = GoldilocksField::from_canonical_usize(messages.len());

        // Build the stark
        let num_rounds = padded_chunks_values.len();
        let mut builder = BytesBuilder::<BLAKE2BTest>::new();
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<U64Register>(16))
            .collect::<Vec<_>>();
        let t_values = builder.alloc_array_public::<U64Register>(num_rounds);
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let digest_indices = builder.alloc_array_public(messages.len());
        let num_messages = builder.alloc_public();
        let hash_state = builder.blake2b(
            &padded_chunks,
            &t_values,
            &end_bits,
            &end_bits,
            &digest_indices,
            &num_messages,
        );

        /*
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);
        */

        let num_rows_degree = log2_ceil(96 * num_rounds);
        let num_rows = 1 << num_rows_degree;
        let stark = builder.build::<C, 2>(num_rows);

        // Write trace.
        let mut writer_data = AirWriterData::new(&stark.air_data, num_rows);
        let mut writer = writer_data.public_writer();

        writer.write(&num_messages, &num_messages_value);
        let mut intial_state = IV;
        for i in 0..num_rounds {
            let padded_chunk = padded_chunks_values[i];
            writer.write_array(&padded_chunks[i], padded_chunk);
            writer.write(&end_bits.get(i), &end_bits_values[i]);
            writer.write(&t_values.get(i), &t_values_values[i]);

            let chunk = padded_chunks_values[i];
            let a = chunk.iter().flatten().collect_vec();
            println!("len of a is {}", a.len());
            let hash = BLAKE2BPure::compress(
                &chunk
                    .iter()
                    .flatten()
                    .map(|x| GoldilocksField::as_canonical_u64(x) as u8)
                    .collect_vec(),
                &mut intial_state,
                0,
                true,
            );

            writer.write_array(
                &hash_state[0],
                hash.map(u64_to_le_field_bytes::<GoldilocksField>),
            );
        }

        for (i, digest_index) in digest_indices_values.iter().enumerate() {
            writer.write(&digest_indices.get(i), digest_index);
        }

        timed!(timing, "write input", {
            stark.air_data.write_global_instructions(&mut writer);

            for mut chunk in writer_data.chunks(num_rows) {
                for i in 0..num_rows {
                    println!("writing trace instructions for row {}", i);
                    let mut writer = chunk.window_writer(i);
                    stark.air_data.write_trace_instructions(&mut writer);
                }
            }
        });

        /*
        // Compare expected digests with the trace values.
        let writer = writer_data.public_writer();
        for (digest, expected) in hash_state.iter().zip_eq(expected_digests) {
            let array: ArrayRegister<S::IntRegister> = (*digest).into();
            let digest = writer
                .read_array::<_, 8>(&array)
                .map(|x| S::field_value_to_int(&x));
            let expected_digest = S::decode(expected);
            assert_eq!(digest, expected_digest);
        }
        */

        let (trace, public) = (writer_data.trace, writer_data.public);

        let proof = timed!(
            timing,
            "generate stark proof",
            stark.prove(&trace, &public, &mut timing).unwrap()
        );

        stark.verify(proof.clone(), &public).unwrap();

        /*
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
        */
    }
}
