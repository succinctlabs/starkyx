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
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::timed;
    use plonky2::util::log2_ceil;
    use plonky2::util::timing::TimingTree;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
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

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct BLAKE2BTest;

    impl AirParameters for BLAKE2BTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;
        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 1323;
        const EXTENDED_COLUMNS: usize = 834;
    }

    #[test]
    pub fn test_blake2b() {
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha", log::Level::Debug);

        let num_rounds = 2;
        let num_messages_value = 1;

        let mut builder = BytesBuilder::<BLAKE2BTest>::new();
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<U64Register>(16))
            .collect::<Vec<_>>();
        let t_values = builder.alloc_array_public::<U64Register>(num_rounds);
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let digest_indices = builder.alloc_array_public(num_messages_value);
        let num_messages = builder.alloc_public();
        let hash_state = builder.blake2b(
            &padded_chunks,
            &t_values,
            &end_bits,
            &end_bits,
            &digest_indices,
            &num_messages,
        );

        let num_rows_degree = log2_ceil(96 * num_rounds);
        let num_rows = 1 << num_rows_degree;
        let stark = builder.build::<C, 2>(num_rows);

        /*
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);
        */

        let message = b"325623465236262asdagds326fdsfy3w456gery46462ialweurnawieyailughoiwabn4bkq23bh2jh5bkwaeublaieunrqi4awijbjkahtiqi3uwagastt3asgesgg3";
        let padded_chunks_values: Vec<[GoldilocksField; 8]> = BLAKE2BUtil::pad(message.as_ref(), 2)
            .chunks_exact(8)
            .map(|x| {
                let a: [GoldilocksField; 8] = x
                    .iter()
                    .map(|y| GoldilocksField::from_canonical_u8(*y))
                    .collect_vec()
                    .as_slice()
                    .try_into()
                    .unwrap();
                a
            })
            .collect_vec();
        let mut t_values_values = [[GoldilocksField::ZERO; 8], [GoldilocksField::ZERO; 8]];
        t_values_values[0][0] = GoldilocksField::from_canonical_u8(128);
        t_values_values[1][0] = GoldilocksField::from_canonical_u8(1);
        let end_bits_values = [GoldilocksField::ZERO, GoldilocksField::ONE];
        let digest_indices_values = [GoldilocksField::ONE];
        let num_messages_value = GoldilocksField::ONE;

        // Write trace.
        let writer = TraceWriter::new(&stark.air_data, num_rows);

        writer.write(&num_messages, &num_messages_value, 0);
        let mut intial_state = IV;
        for i in 0..num_rounds {
            writer.write_array(
                &padded_chunks[i],
                &padded_chunks_values[i * 16..(i + 1) * 16],
                0,
            );
            writer.write(&end_bits.get(i), &end_bits_values[i], 0);
            writer.write(&t_values.get(i), &t_values_values[i], 0);

            let hash = BLAKE2BPure::compress(
                &padded_chunks_values
                    .iter()
                    .flatten()
                    .map(|x| GoldilocksField::as_canonical_u64(&x) as u8)
                    .collect_vec(),
                &mut intial_state,
                0,
                true,
            );

            writer.write_array(
                &hash_state[0],
                hash.map(u64_to_le_field_bytes::<GoldilocksField>),
                0,
            );
        }

        for i in 0..num_messages_value.as_canonical_u64() as usize {
            writer.write(&digest_indices.get(i), &digest_indices_values[i], 0);
        }

        writer.write_global_instructions(&stark.air_data);
        println!("wrote global instructions");
        (0..num_rounds).for_each(|r| {
            for k in 0..96 {
                let i = r * 96 + k;
                println!("writing row instructions for row {}", i);
                writer.write_row_instructions(&stark.air_data, i);
                println!("wrote row instructions for row {}", i);
            }
        });

        /*
        // Compare expected digests with the trace values.
        for (digest, expected) in hash_state.iter().zip_eq(expected_digests) {
            let array: ArrayRegister<S::IntRegister> = (*digest).into();
            let digest = writer
                .read_array::<_, 8>(&array, 0)
                .map(|x| S::field_value_to_int(&x));
            let expected_digest = S::decode(expected);
            assert_eq!(digest, expected_digest);
        }
        */

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
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
