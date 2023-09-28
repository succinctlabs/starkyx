use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, Read, Write};
use serde::{Deserialize, Serialize};

use super::{
    U64Value, HASH_ARRAY_SIZE, INITIAL_HASH, INITIAL_HASH_COMPRESS, INVERSION_CONST,
    MSG_ARRAY_SIZE, NUM_MIX_ROUNDS,
};
use crate::chip::builder::AirBuilder;
use crate::chip::hash::blake::blake2b::BLAKE2BGadget;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::uint::bytes::lookup_table::table::ByteLookupTable;
use crate::chip::uint::operations::instruction::U32Instruction;
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::{AirParameters, Chip};
use crate::math::field::PrimeField64;
use crate::math::prelude::CubicParameters;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::proof::StarkProofTarget;
use crate::plonky2::stark::prover::StarkyProver;
use crate::plonky2::stark::verifier::set_stark_proof_target;
use crate::plonky2::stark::Starky;
use crate::utils::serde::{BufferRead, BufferWrite};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BLAKE2BAirParameters<F, E>(pub PhantomData<(F, E)>);

impl<F: PrimeField64, E: CubicParameters<F>> AirParameters for BLAKE2BAirParameters<F, E> {
    type Field = F;
    type CubicParams = E;

    type Instruction = U32Instruction;

    const NUM_FREE_COLUMNS: usize = 3541;
    const EXTENDED_COLUMNS: usize = 1617;
    const NUM_ARITHMETIC_COLUMNS: usize = 0;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BLAKE2BGenerator<
    F: PrimeField64,
    E: CubicParameters<F>,
    C,
    const D: usize,
    L: AirParameters + 'static + Clone + Debug + Send + Sync,
> {
    pub padded_messages: Vec<Target>,
    pub msg_lens: Vec<Target>,
    pub pub_values_target: BLAKE2BPublicData<Target>,
    pub config: StarkyConfig<C, D>,
    pub proof_target: StarkProofTarget<D>,
    pub _phantom: PhantomData<(F, E, L)>,
}

pub struct BLAKE2BStarkData<F: PrimeField64, E: CubicParameters<F>, C, const D: usize> {
    pub stark: Starky<Chip<BLAKE2BAirParameters<F, E>>>,
    pub table: ByteLookupTable,
    pub trace_generator: ArithmeticGenerator<BLAKE2BAirParameters<F, E>>,
    pub config: StarkyConfig<C, D>,
    pub gadget: BLAKE2BGadget,
}

impl<
        F: PrimeField64,
        E: CubicParameters<F>,
        C,
        const D: usize,
        L: AirParameters + 'static + Clone + Debug + Send + Sync,
    > BLAKE2BGenerator<F, E, C, D, L>
{
    pub fn id() -> String {
        "BLAKE2BGenerator".to_string()
    }

    pub fn stark_data() -> BLAKE2BStarkData<F, E, C, D>
    where
        F: RichField + Extendable<D>,
        C: CurtaConfig<D, F = F>,
        E: CubicParameters<F>,
    {
        let mut air_builder = AirBuilder::<BLAKE2BAirParameters<F, E>>::new();
        let clk = air_builder.clock();

        let (mut operations, table) = air_builder.byte_operations();

        let mut bus = air_builder.new_bus();
        let channel_idx = bus.new_channel(&mut air_builder);

        let gadget = air_builder.process_blake2b(&clk, &mut bus, channel_idx, &mut operations);

        air_builder.register_byte_lookup(operations, &table);
        air_builder.constrain_bus(bus);

        let (air, trace_data) = air_builder.build();

        let stark = Starky::new(air);
        let num_rows = 1 << 16;
        let config = StarkyConfig::<C, D>::standard_fast_config(num_rows);

        let trace_generator =
            ArithmeticGenerator::<BLAKE2BAirParameters<F, E>>::new(trace_data, num_rows);

        BLAKE2BStarkData {
            stark,
            table,
            trace_generator,
            config,
            gadget,
        }
    }
}

impl<
        F: RichField + Extendable<D>,
        E: CubicParameters<F>,
        C: CurtaConfig<D, F = F>,
        const D: usize,
        L: AirParameters + 'static + Clone + Debug + Send + Sync,
    > SimpleGenerator<F, D> for BLAKE2BGenerator<F, E, C, D, L>
{
    fn id(&self) -> String {
        Self::id()
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        let data = bincode::serialize(self).unwrap();
        dst.write_bytes(&data)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self>
    where
        Self: Sized,
    {
        let bytes = src.read_bytes()?;
        let data = bincode::deserialize(&bytes).unwrap();
        Ok(data)
    }

    fn dependencies(&self) -> Vec<Target> {
        let mut dependencies = Vec::new();
        dependencies.extend_from_slice(self.padded_messages.as_slice());
        dependencies.extend_from_slice(self.msg_lens.as_slice());
        dependencies
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let BLAKE2BStarkData {
            stark,
            table,
            trace_generator,
            config,
            gadget,
        } = Self::stark_data();

        let padded_messages = self
            .padded_messages
            .iter()
            .map(|x| witness.get_target(*x).as_canonical_u64() as u8)
            .collect::<Vec<_>>();

        let num_rows = 1 << 16;
        let max_num_chunks = num_rows / 128;
        assert!(padded_messages.len() <= max_num_chunks * 128);

        let msg_sizes = self
            .msg_lens
            .iter()
            .map(|x| witness.get_target(*x).as_canonical_u64())
            .collect::<Vec<_>>();

        let message_chunks = msg_sizes.iter().scan(0, |idx, size| {
            let mut num_chunks = *size as usize / 128;

            if (*size % 128 != 0) || (*size == 0) {
                num_chunks += 1;
            }

            let chunk = padded_messages[*idx as usize..*idx as usize + 128 * num_chunks].to_vec();
            *idx += 128 * size;
            Some(chunk)
        });

        // Write trace values
        let writer = trace_generator.new_writer();
        table.write_table_entries(&writer);
        let blake_public_values = gadget.write(message_chunks, &msg_sizes, &writer, num_rows);

        for i in 0..num_rows {
            writer.write_row_instructions(&trace_generator.air_data, i);
        }

        // Fill blake2b public values into the output buffer
        self.pub_values_target
            .set_targets(blake_public_values, out_buffer);

        let public_inputs: Vec<_> = writer.public.read().unwrap().clone();

        let proof =
            StarkyProver::<F, C, D>::prove(&config, &stark, &trace_generator, &public_inputs)
                .unwrap();

        set_stark_proof_target(out_buffer, &self.proof_target, &proof);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BLAKE2BPublicData<T> {
    pub msg_chunks: Vec<U64Value<T>>,
    pub t: Vec<U64Value<T>>,
    pub msg_last_chunk: Vec<T>,
    //pub max_chunk: Vec<T>,
    pub hash_state: Vec<U64Value<T>>,
}

impl BLAKE2BPublicData<Target> {
    pub fn add_virtual<
        F: RichField + Extendable<D>,
        const D: usize,
        L: AirParameters + 'static + Clone + Debug + Send + Sync,
    >(
        builder: &mut CircuitBuilder<F, D>,
        digests: &[Target],
        chunk_sizes: &[usize],
    ) -> Self {
        let num_rows = 1 << 16;
        let num_chunks = num_rows / NUM_MIX_ROUNDS;

        let msg_chunks_targets = (0..num_chunks * MSG_ARRAY_SIZE)
            .map(|_| builder.add_virtual_target_arr::<8>())
            .collect::<Vec<_>>();

        let msg_last_chunk_targets = (0..num_chunks)
            .map(|_| builder.add_virtual_target())
            .collect::<Vec<_>>();

        // let max_chunk_targets = (0..num_chunks)
        //     .map(|_| builder.add_virtual_target())
        //     .collect::<Vec<_>>();

        let t_targets = (0..num_chunks)
            .map(|_| builder.add_virtual_target_arr::<8>())
            .collect::<Vec<_>>();

        let mut hash_state_targets = Vec::new();
        assert!(digests.len() / 8 <= num_chunks * HASH_ARRAY_SIZE);
        assert!(digests.len() % 8 == 0);

        for (digest, chunk_size) in digests.chunks_exact(32).zip_eq(chunk_sizes.iter()) {
            hash_state_targets
                .extend((0..8 * (chunk_size - 1)).map(|_| builder.add_virtual_target_arr::<8>()));

            let u64_digest_byte = digest.chunks_exact(8).map(|arr| {
                let array: [Target; 8] = arr.try_into().unwrap();
                array
            });
            hash_state_targets.extend(u64_digest_byte);
            hash_state_targets.extend((0..4).map(|_| builder.add_virtual_target_arr::<8>()));
        }

        for _ in hash_state_targets.len()..num_chunks * HASH_ARRAY_SIZE {
            hash_state_targets.push(builder.add_virtual_target_arr::<8>());
        }

        BLAKE2BPublicData {
            msg_chunks: msg_chunks_targets,
            t: t_targets,
            msg_last_chunk: msg_last_chunk_targets,
            //max_chunk: max_chunk_targets,
            hash_state: hash_state_targets,
        }
    }

    pub fn set_targets<F: RichField>(
        &self,
        values: BLAKE2BPublicData<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) {
        assert!(
            self.msg_chunks.len() == values.msg_chunks.len(),
            "msg_chunks length mismatch"
        );
        for (pub_msg_chunk_target, pub_msg_chunk_value) in
            self.msg_chunks.iter().zip_eq(values.msg_chunks.iter())
        {
            out_buffer.set_target_arr(pub_msg_chunk_target, pub_msg_chunk_value);
        }

        assert!(self.t.len() == values.t.len(), "t length mismatch");
        for (pub_t_target, pub_t_value) in self.t.iter().zip_eq(values.t.iter()) {
            out_buffer.set_target_arr(pub_t_target, pub_t_value);
        }

        assert!(
            self.msg_last_chunk.len() == values.msg_last_chunk.len(),
            "last_msg_last_chunkchunk_bit length mismatch"
        );
        for (pub_last_chunk_bit_target, pub_last_chunk_bit_value) in self
            .msg_last_chunk
            .iter()
            .zip_eq(values.msg_last_chunk.iter())
        {
            out_buffer.set_target(*pub_last_chunk_bit_target, *pub_last_chunk_bit_value);
        }

        assert!(
            self.hash_state.len() == values.hash_state.len(),
            "hash_state length mismatch"
        );
        for (hash_target, hash_value) in self.hash_state.iter().zip_eq(values.hash_state.iter()) {
            out_buffer.set_target_arr(hash_target, hash_value);
        }
    }

    pub fn public_input_targets<F: RichField + Extendable<D>, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Vec<Target> {
        INITIAL_HASH
            .map(|value| u64_to_le_field_bytes(value).map(|x| builder.constant(x)))
            .into_iter()
            .flatten()
            .chain(
                INITIAL_HASH_COMPRESS
                    .map(|value| u64_to_le_field_bytes(value).map(|x| builder.constant(x)))
                    .into_iter()
                    .flatten(),
            )
            .chain(u64_to_le_field_bytes(INVERSION_CONST).map(|x| builder.constant(x)))
            .chain(self.msg_chunks.iter().flatten().copied())
            .chain(self.t.iter().flatten().copied())
            .chain(self.msg_last_chunk.clone())
            .chain(self.hash_state.iter().flatten().copied())
            .collect()
    }
}

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

        let mut state: [u64; 8] = [0; 8];
        state[..8].copy_from_slice(&INITIAL_HASH[..8]);

        let num_chunks = padded_message.len() / 128;
        let mut bytes_compressed = 0u64;
        assert!(padded_message.len() % 128 == 0);
        for (chunk_num, chunk) in padded_message.chunks_exact(128).enumerate() {
            let last_chunk = chunk_num == num_chunks - 1;

            if last_chunk {
                bytes_compressed = message_len as u64;
            } else {
                bytes_compressed += 128;
            }

            state = BLAKE2BGadget::compress(chunk, &mut state, bytes_compressed, last_chunk);
        }

        // We only support a digest of 32 bytes.  Retrieve the first four elements of the state
        let binding = state[0..4]
            .iter()
            .flat_map(|x| u64_to_le_field_bytes::<F>(*x))
            .collect::<Vec<_>>();
        let digest_bytes = binding.as_slice();

        out_buffer.set_target_arr(&self.digest_bytes, digest_bytes);
    }
}
