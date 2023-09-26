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

use super::{SHA256Gadget, SHA256PublicData, INITIAL_HASH, ROUND_CONSTANTS};
use crate::chip::builder::AirBuilder;
use crate::chip::register::Register;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::uint::bytes::lookup_table::table::ByteLookupTable;
use crate::chip::uint::operations::instruction::U32Instruction;
use crate::chip::uint::register::U32Register;
use crate::chip::uint::util::u32_to_le_field_bytes;
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::{CubicParameters, *};
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::proof::StarkProofTarget;
use crate::plonky2::stark::prover::StarkyProver;
use crate::plonky2::stark::verifier::set_stark_proof_target;
use crate::plonky2::stark::Starky;
use crate::utils::serde::{BufferRead, BufferWrite};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SHA256AirParameters<F, E>(pub PhantomData<(F, E)>);

pub type U32Target = <U32Register as Register>::Value<Target>;

pub const SHA256_COLUMNS: usize = 551 + 927;

#[derive(Debug, Clone)]
pub struct MessageChunks {
    pub values: Vec<u8>,
    pub chunk_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct SHA256HintGenerator {
    padded_message: Vec<Target>,
    digest_bytes: [Target; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SHA256Generator<F: PrimeField64, E: CubicParameters<F>, C, const D: usize> {
    pub padded_messages: Vec<Target>,
    pub chunk_sizes: Vec<usize>,
    pub pub_values_target: SHA256PublicData<Target>,
    pub config: StarkyConfig<C, D>,
    pub proof_target: StarkProofTarget<D>,
    pub _marker: PhantomData<(F, E)>,
}

pub struct SHA256StarkData<F: PrimeField64, E: CubicParameters<F>, C, const D: usize> {
    pub stark: Starky<Chip<SHA256AirParameters<F, E>>>,
    pub table: ByteLookupTable,
    pub trace_generator: ArithmeticGenerator<SHA256AirParameters<F, E>>,
    pub config: StarkyConfig<C, D>,
    pub gadget: SHA256Gadget,
}

impl<F: PrimeField64, E: CubicParameters<F>> AirParameters for SHA256AirParameters<F, E> {
    type Field = F;
    type CubicParams = E;

    type Instruction = U32Instruction;

    const NUM_FREE_COLUMNS: usize = 551;
    const EXTENDED_COLUMNS: usize = 927;
    const NUM_ARITHMETIC_COLUMNS: usize = 0;
}

impl<F: PrimeField64, E: CubicParameters<F>, C, const D: usize> SHA256Generator<F, E, C, D> {
    pub fn id() -> String {
        "SHA256Generator".to_string()
    }

    pub fn stark_data() -> SHA256StarkData<F, E, C, D>
    where
        F: RichField + Extendable<D>,
        C: CurtaConfig<D, F = F>,
        E: CubicParameters<F>,
    {
        let mut air_builder = AirBuilder::<SHA256AirParameters<F, E>>::new();
        let clk = air_builder.clock();

        let (mut operations, table) = air_builder.byte_operations();

        let mut bus = air_builder.new_bus();
        let channel_idx = bus.new_channel(&mut air_builder);

        let gadget =
            air_builder.process_sha_256_batch(&clk, &mut bus, channel_idx, &mut operations);

        air_builder.register_byte_lookup(operations, &table);
        air_builder.constrain_bus(bus);

        let (air, trace_data) = air_builder.build();

        let num_rows = 1 << 16;
        let stark = Starky::new(air);
        let config = StarkyConfig::<C, D>::standard_fast_config(num_rows);

        let trace_generator =
            ArithmeticGenerator::<SHA256AirParameters<F, E>>::new(trace_data, num_rows);

        SHA256StarkData {
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
        C: CurtaConfig<D, F = F>,
        E: CubicParameters<F>,
        const D: usize,
    > SimpleGenerator<F, D> for SHA256Generator<F, E, C, D>
{
    fn id(&self) -> String {
        Self::id()
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        let data = bincode::serialize(self).unwrap();
        dst.write_bytes(&data)
    }

    fn deserialize(
        src: &mut Buffer,
        _: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self>
    where
        Self: Sized,
    {
        let bytes = src.read_bytes()?;
        let data = bincode::deserialize(&bytes).unwrap();
        Ok(data)
    }

    fn dependencies(&self) -> Vec<Target> {
        self.padded_messages.clone()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let SHA256StarkData {
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
        assert_eq!(padded_messages.len(), 1024 * 64);

        let message_chunks = self.chunk_sizes.iter().scan(0, |idx, size| {
            let chunk = padded_messages[*idx..*idx + 64 * size].to_vec();
            *idx += 64 * size;
            Some(chunk)
        });

        // Write trace values
        let num_rows = 1 << 16;
        let writer = trace_generator.new_writer();
        table.write_table_entries(&writer);
        let sha_public_values = gadget.write(message_chunks, &writer);
        for i in 0..num_rows {
            writer.write_row_instructions(&trace_generator.air_data, i);
        }
        table.write_multiplicities(&writer);

        // Fill sha public values into the output buffer
        self.pub_values_target
            .set_targets(sha_public_values, out_buffer);

        let public_inputs: Vec<_> = writer.public.read().unwrap().clone();

        let proof =
            StarkyProver::<F, C, D>::prove(&config, &stark, &trace_generator, &public_inputs)
                .unwrap();

        set_stark_proof_target(out_buffer, &self.proof_target, &proof);
    }
}

impl SHA256PublicData<Target> {
    pub fn add_virtual<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        digests: &[Target],
        chunk_sizes: &[usize],
    ) -> Self {
        let public_w_targets = (0..16 * 1024)
            .map(|_| builder.add_virtual_target_arr::<4>())
            .collect::<Vec<_>>();

        // let end_bits_targets = builder.add_virtual_targets(1024);
        let mut end_bits_targets = Vec::new();
        let mut hash_state_targets = Vec::new();

        for (digest, chunk_size) in digests.chunks_exact(32).zip_eq(chunk_sizes.iter()) {
            end_bits_targets.extend((0..(chunk_size - 1)).map(|_| builder.zero()));
            end_bits_targets.push(builder.one());

            hash_state_targets
                .extend((0..8 * (chunk_size - 1)).map(|_| builder.add_virtual_target_arr::<4>()));

            // Convert digest to little endian u32 chunks
            let u32_digest = digest.chunks_exact(4).map(|arr| {
                let mut array: [Target; 4] = arr.try_into().unwrap();
                array.reverse();
                array
            });
            hash_state_targets.extend(u32_digest);
        }

        SHA256PublicData {
            public_w: public_w_targets,
            hash_state: hash_state_targets,
            end_bits: end_bits_targets,
        }
    }

    pub fn set_targets<F: RichField>(
        &self,
        values: SHA256PublicData<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) {
        for (pub_w_target, pub_w_value) in self.public_w.iter().zip_eq(values.public_w.iter()) {
            out_buffer.set_target_arr(pub_w_target, pub_w_value);
        }
        for (hash_target, hash_value) in self.hash_state.iter().zip_eq(values.hash_state.iter()) {
            out_buffer.set_target_arr(hash_target, hash_value);
        }
    }

    pub fn public_input_targets<F: RichField + Extendable<D>, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Vec<Target> {
        self.public_w
            .iter()
            .flatten()
            .copied()
            .chain(
                INITIAL_HASH
                    .map(|value| u32_to_le_field_bytes(value).map(|x| builder.constant(x)))
                    .into_iter()
                    .flatten(),
            )
            .chain(
                ROUND_CONSTANTS
                    .map(|value| u32_to_le_field_bytes(value).map(|x| builder.constant(x)))
                    .into_iter()
                    .flatten(),
            )
            .chain(self.hash_state.iter().flatten().copied())
            .chain(self.end_bits.iter().copied())
            .collect()
    }
}

impl SHA256HintGenerator {
    pub fn new(padded_message: &[Target], digest_bytes: [Target; 32]) -> Self {
        SHA256HintGenerator {
            padded_message: padded_message.to_vec(),
            digest_bytes,
        }
    }
}

impl SHA256HintGenerator {
    pub fn id() -> String {
        "SHA256HintGenerator".to_string()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for SHA256HintGenerator {
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
        let digest_bytes = src.read_target_vec()?;
        Ok(Self {
            padded_message,
            digest_bytes: digest_bytes.try_into().unwrap(),
        })
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let padded_message = witness
            .get_targets(&self.padded_message)
            .into_iter()
            .map(|x| x.as_canonical_u64() as u8)
            .collect::<Vec<_>>();

        let mut state = INITIAL_HASH;
        for chunk in padded_message.chunks_exact(64) {
            let w_val = SHA256Gadget::process_inputs(chunk);
            state = SHA256Gadget::compress_round(state, &w_val, ROUND_CONSTANTS);
        }

        let digest_bytes = state
            .map(|x| {
                let mut arr = u32_to_le_field_bytes::<F>(x);
                arr.reverse();
                arr
            })
            .concat();

        out_buffer.set_target_arr(&self.digest_bytes, &digest_bytes);
    }
}
