use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, Read, Write};

use super::INITIAL_HASH;
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::math::field::PrimeField64;

#[derive(Debug, Clone)]
pub struct BLAKE2BHintGenerator {
    padded_message: Vec<Target>,
    message_len: Target,
    digest_bytes: [Target; 32],
}

impl BLAKE2BHintGenerator {
    pub fn new(padded_message: &[Target], message_len: Target, digest_bytes: [Target; 32]) -> Self {
        BLAKE2BHintGenerator {
            padded_message: padded_message.to_vec(),
            message_len,
            digest_bytes,
        }
    }
}

impl BLAKE2BHintGenerator {
    pub fn id() -> String {
        "BLAKE2BHintGenerator".to_string()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for BLAKE2BHintGenerator {
    fn id(&self) -> String {
        Self::id()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.padded_message.clone()
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        dst.write_target_vec(&self.padded_message)?;
        dst.write_target(self.message_len)?;
        dst.write_target_vec(&self.digest_bytes)?;
        Ok(())
    }

    fn deserialize(
        src: &mut Buffer,
        _: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self>
    where
        Self: Sized,
    {
        let padded_message = src.read_target_vec()?;
        let message_len = src.read_target()?;
        let digest_bytes = src.read_target_vec()?;
        Ok(Self {
            padded_message,
            message_len,
            digest_bytes: digest_bytes.try_into().unwrap(),
        })
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let padded_message = witness
            .get_targets(&self.padded_message)
            .into_iter()
            .map(|x| x.as_canonical_u64() as u8)
            .collect::<Vec<_>>();

        let message_len = witness.get_target(self.message_len).as_canonical_u64() as usize;

        let mut state = &INITIAL_HASH[0..8];
        let num_chunks = padded_message.len() / 128;
        let mut bytes_compressed = 0;
        assert!(padded_message.len() % 128 == 0);
        for (chunk_num, chunk) in padded_message.chunks_exact(128).enumerate() {
            let last_chunk = chunk_num == num_chunks - 1;

            if last_chunk {
                bytes_compressed = message_len;
            } else {
                bytes_compressed += 128;
            }

            state = BLAKE2BGadget::compress_round(chunk, state, bytes_compressed, last_chunk);
        }

        // We only support a digest of 32 bytes.  Retrieve the first four elements of the state
        let digest_bytes = state[0..4]
            .iter()
            .map(|x| {
                let mut arr = u64_to_le_field_bytes::<F>(*x);
                arr.reverse();
                arr
            })
            .flatten()
            .collect_vec()
            .as_slice();

        out_buffer.set_target_arr(&self.digest_bytes, &digest_bytes);
    }
}
