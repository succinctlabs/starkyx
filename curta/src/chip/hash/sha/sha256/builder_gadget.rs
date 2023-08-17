use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use super::generator::{SHA256AirParameters, SHA256Generator, SHA256_COLUMNS};
use super::SHA256PublicData;
use crate::chip::builder::AirBuilder;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::AirParameters;
use crate::math::prelude::CubicParameters;
use crate::plonky2::stark::config::StarkyConfig;
use crate::plonky2::stark::gadget::StarkGadget;
use crate::plonky2::stark::generator::simple::SimpleStarkWitnessGenerator;
use crate::plonky2::stark::Starky;

#[derive(Debug, Clone, Copy)]
pub struct CurtaU32Array<const N: usize>(pub [Target; N]);

#[derive(Debug, Clone)]
pub struct SHA256BuilderGadget<F, E, const D: usize> {
    pub padded_messages: Vec<Target>,
    chunk_sizes: Vec<Target>,
    _marker: PhantomData<(F, E)>,
}

pub trait SHA256Builder<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> {
    type Gadget;

    fn init_sha_256(&mut self) -> Self::Gadget;

    fn sha_256<const N: usize>(
        &mut self,
        padded_message: &CurtaU32Array<N>,
        gadget: &mut Self::Gadget,
    );

    fn constrain_sha_256_gadget<C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone>(
        &mut self,
        gadget: Self::Gadget,
    ) where
        C::Hasher: AlgebraicHasher<F>;
}

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> SHA256Builder<F, E, D>
    for CircuitBuilder<F, D>
{
    type Gadget = SHA256BuilderGadget<F, E, D>;

    fn init_sha_256(&mut self) -> Self::Gadget {
        SHA256BuilderGadget {
            padded_messages: Vec::new(),
            chunk_sizes: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn sha_256<const N: usize>(
        &mut self,
        padded_message: &CurtaU32Array<N>,
        gadget: &mut Self::Gadget,
    ) {
        gadget.padded_messages.extend_from_slice(&padded_message.0);
        gadget.chunk_sizes.push(self.constant(F::ONE));
    }

    fn constrain_sha_256_gadget<C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone>(
        &mut self,
        gadget: Self::Gadget,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        // Allocate public input targets
        let public_sha_targets = SHA256PublicData::add_virtual(self);

        // Make the air
        let mut air_builder = AirBuilder::<SHA256AirParameters<F, E>>::new();
        let clk = air_builder.clock();

        let (mut operations, table) = air_builder.byte_operations();

        let mut bus = air_builder.new_bus();
        let channel_idx = bus.new_channel(&mut air_builder);

        let sha_gadget =
            air_builder.process_sha_256_batch(&clk, &mut bus, channel_idx, &mut operations);

        air_builder.register_byte_lookup(operations, &table);
        air_builder.constrain_bus(bus);

        let (air, trace_data) = air_builder.build();

        let generator = ArithmeticGenerator::<SHA256AirParameters<F, E>>::new(trace_data);

        let public_input_target = public_sha_targets.public_input_targets(self);

        let sha_generator = SHA256Generator {
            gadget: sha_gadget,
            table,
            padded_messages: gadget.padded_messages,
            chunk_sizes: gadget.chunk_sizes,
            trace_generator: generator.clone(),
            pub_values_target: public_sha_targets,
        };

        self.add_simple_generator(sha_generator);

        let stark = Starky::<_, SHA256_COLUMNS>::new(air);
        let config =
            StarkyConfig::<F, C, D>::standard_fast_config(SHA256AirParameters::<F, E>::num_rows());
        let virtual_proof = self.add_virtual_stark_proof(&stark, &config);
        self.verify_stark_proof(&config, &stark, virtual_proof.clone(), &public_input_target);

        let stark_generator = SimpleStarkWitnessGenerator::new(
            config,
            stark,
            virtual_proof,
            public_input_target,
            generator,
        );

        self.add_simple_generator(stark_generator);
    }
}

#[cfg(test)]
mod tests {

    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use subtle_encoding::hex::decode;

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::hash::sha::sha256::SHA256Gadget;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::uint::util::u32_to_le_field_bytes;
    use crate::chip::AirParameters;

    #[test]
    fn test_sha_256_plonky_gadget() {
        type F = GoldilocksField;
        type E = GoldilocksCubicParameters;
        type C = PoseidonGoldilocksConfig;
        const D: usize = 2;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Sha256 Plonky2 gadget test", log::Level::Debug);

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let mut gadget: SHA256BuilderGadget<F, E, D> = builder.init_sha_256();

        let padded_msg_targets = (0..1024)
            .map(|_| CurtaU32Array(builder.add_virtual_target_arr::<16>()))
            .collect::<Vec<_>>();

        for padded_msg in padded_msg_targets.iter() {
            builder.sha_256(padded_msg, &mut gadget);
        }

        builder.constrain_sha_256_gadget::<C>(gadget);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();

        let short_msg_1 = decode("").unwrap();
        let expected_digest_1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        // let short_msg_1 = b"plonky2".to_vec();
        // let expected_digest = "8943a85083f16e93dc92d6af455841daacdae5081aa3125b614a626df15461eb";

        let short_msg_2 = b"abc".to_vec();
        let expected_digest_2 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let long_msg = decode("243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917").unwrap();
        let expected_digest_long =
            "aca16131a2e4c4c49e656d35aac1f0e689b3151bb108fa6cf5bcc3ac08a09bf9";

        let messages = (0..1024).map(|_| short_msg_1.clone()).collect::<Vec<_>>();
        let padded_messages = messages
            .iter()
            .map(|m| {
                SHA256Gadget::pad(m)
                    .into_iter()
                    .map(F::from_canonical_u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // let expected_digests: Vec<[u32; 8]> = (0..256)
        //     .flat_map(|_| {
        //         [
        //             expected_digest_1.clone(),
        //             expected_digest_long.clone(),
        //             expected_digest_2.clone(),
        //         ]
        //     })
        //     .map(|digest| {
        //         hex::decode(digest)
        //             .unwrap()
        //             .chunks_exact(4)
        //             .map(|x| u32::from_be_bytes(x.try_into().unwrap()))
        //             .collect::<Vec<_>>()
        //             .try_into()
        //             .unwrap()
        //     })
        //     .collect::<Vec<_>>();
        // assert_eq!(expected_digests.len(), padded_messages.len());

        // let mut digest_iter = expected_digests.into_iter();

        for (msg_target, msg) in padded_msg_targets.iter().zip(padded_messages.iter()) {
            pw.set_target_arr(&msg_target.0, msg);
        }

        let recursive_proof = timed!(
            timing,
            "Generate proof",
            plonky2::plonk::prover::prove(&data.prover_only, &data.common, pw, &mut timing)
        )
        .unwrap();
        timing.print();
        data.verify(recursive_proof).unwrap();
    }
}
