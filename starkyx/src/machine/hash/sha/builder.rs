use super::algorithm::SHAir;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::machine::builder::Builder;

pub trait SHABuilder: Builder {
    fn sha<S: SHAir<Self, CYCLE_LENGTH>, const CYCLE_LENGTH: usize>(
        &mut self,
        padded_chunks: &[ArrayRegister<S::IntRegister>],
        end_bits: &ArrayRegister<BitRegister>,
        digest_bits: &ArrayRegister<BitRegister>,
        digest_indices: ArrayRegister<ElementRegister>,
    ) -> Vec<S::StateVariable> {
        S::sha(self, padded_chunks, end_bits, digest_bits, digest_indices)
    }
}

impl<B: Builder> SHABuilder for B {}

#[cfg(test)]
pub mod test_utils {
    use core::fmt::Debug;

    use itertools::Itertools;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::timed;
    use plonky2::util::log2_ceil;
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::chip::trace::writer::data::AirWriterData;
    use crate::chip::trace::writer::AirWriter;
    use crate::chip::uint::operations::instruction::UintInstructions;
    use crate::chip::{AirParameters, Chip};
    use crate::machine::bytes::builder::BytesBuilder;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::math::prelude::*;
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};
    use crate::plonky2::Plonky2Air;

    pub fn test_sha<
        'a,
        L,
        S,
        I: IntoIterator<Item = &'a [u8]>,
        J: IntoIterator<Item = &'a str>,
        const CYCLE_LENGTH: usize,
    >(
        messages: I,
        expected_digests: J,
    ) where
        L: AirParameters<Field = GoldilocksField, CubicParams = GoldilocksCubicParameters>,
        L::Instruction: UintInstructions,
        S: SHAir<BytesBuilder<L>, CYCLE_LENGTH>,
        Chip<L>: Plonky2Air<GoldilocksField, 2>,
        S::Integer: PartialEq + Eq + Debug,
    {
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let mut end_bits_values = Vec::new();
        let mut num_messages = 0;
        let padded_chunks_values = messages
            .into_iter()
            .flat_map(|msg| {
                num_messages += 1;
                let padded_msg = S::pad(msg);
                let num_chunks = padded_msg.len() / 16;
                end_bits_values.extend_from_slice(&vec![GoldilocksField::ZERO; num_chunks - 1]);
                end_bits_values.push(GoldilocksField::ONE);
                padded_msg
            })
            .collect::<Vec<_>>();

        assert_eq!(end_bits_values.len() * 16, padded_chunks_values.len());
        let num_rounds = end_bits_values.len();
        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("test_sha", log::Level::Debug);

        // Build the stark.
        let mut builder = BytesBuilder::<L>::new();
        let padded_chunks = (0..num_rounds)
            .map(|_| builder.alloc_array_public::<S::IntRegister>(16))
            .collect::<Vec<_>>();
        let end_bits = builder.alloc_array_public::<BitRegister>(num_rounds);
        let digest_indices = builder.alloc_array_public(num_messages);
        let hash_state =
            builder.sha::<S, CYCLE_LENGTH>(&padded_chunks, &end_bits, &end_bits, digest_indices);

        let num_rows_degree = log2_ceil(CYCLE_LENGTH * num_rounds);
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
        let mut writer_data = AirWriterData::new(&stark.air_data, num_rows);
        let mut writer = writer_data.public_writer();

        let mut current_state = S::INITIAL_HASH;
        let mut hash_iter = hash_state.iter();
        let mut digest_indices_iter = digest_indices.iter();
        for (i, (((message, register), end_bit), end_bit_value)) in padded_chunks_values
            .chunks_exact(16)
            .zip_eq(padded_chunks.iter())
            .zip_eq(end_bits.iter())
            .zip_eq(end_bits_values.iter())
            .enumerate()
        {
            writer.write_array(register, message.iter().map(|x| S::int_to_field_value(*x)));

            let pre_processed = S::pre_process(message);
            current_state = S::process(current_state, &pre_processed);
            let state = current_state.map(S::int_to_field_value);
            if *end_bit_value == GoldilocksField::ONE {
                writer.write(
                    &digest_indices_iter.next().unwrap(),
                    &GoldilocksField::from_canonical_usize(i),
                );
                let h: S::StateVariable = *hash_iter.next().unwrap();
                let array: ArrayRegister<_> = h.into();
                writer.write_array(&array, &state);
                current_state = S::INITIAL_HASH;
            }

            writer.write(&end_bit, end_bit_value);
        }

        timed!(timing, "write input", {
            stark.air_data.write_global_instructions(&mut writer);

            for mut chunk in writer_data.chunks(num_rows) {
                for i in 0..num_rows {
                    let mut writer = chunk.window_writer(i);
                    stark.air_data.write_trace_instructions(&mut writer);
                }
            }
        });

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

        let (trace, public) = (writer_data.trace, writer_data.public);

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
