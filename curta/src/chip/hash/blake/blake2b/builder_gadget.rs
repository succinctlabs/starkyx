use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use super::generator::BLAKE2BHintGenerator;
use super::BLAKE2BPublicData;
use crate::chip::builder::AirBuilder;
use crate::chip::hash::sha::sha256::generator::SHA256AirParameters;
use crate::chip::hash::CurtaBytes;
use crate::math::prelude::CubicParameters;
use crate::plonky2::stark::config::StarkyConfig;
use crate::plonky2::stark::gadget::StarkGadget;
use crate::plonky2::stark::generator::simple::SimpleStarkWitnessGenerator;

#[derive(Debug, Clone)]
pub struct BLAKE2BBuilderGadget<F, E, const D: usize> {
    pub padded_message: Vec<Target>,
    pub digest: Vec<Target>,
    _marker: PhantomData<(F, E)>,
}

pub trait BLAKE2BBuilder<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> {
    type Gadget;

    fn init_blake2b(&mut self) -> Self::Gadget;

    fn blake2b<const N: usize>(
        &mut self,
        padded_message: &CurtaBytes<N>,
        message_len: Target,
        gadget: &mut Self::Gadget,
    ) -> CurtaBytes<32>;

    /*
    fn constrain_blake2b_gadget<C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone>(
        &mut self,
        gadget: Self::Gadget,
    ) where
        C::Hasher: AlgebraicHasher<F>;
        */
}

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> BLAKE2BBuilder<F, E, D>
    for CircuitBuilder<F, D>
{
    type Gadget = BLAKE2BBuilderGadget<F, E, D>;

    fn init_blake2b(&mut self) -> Self::Gadget {
        BLAKE2BBuilderGadget {
            padded_message: Vec::new(),
            digest: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn blake2b<const N: usize>(
        &mut self,
        padded_message: &CurtaBytes<N>,
        message_len: Target,
        gadget: &mut Self::Gadget,
    ) -> CurtaBytes<32> {
        gadget.padded_message.extend_from_slice(&padded_message.0);
        let digest_bytes = self.add_virtual_target_arr::<32>();
        let hint = BLAKE2BHintGenerator::new(&padded_message.0, message_len, digest_bytes);
        self.add_simple_generator(hint);
        gadget.digest.extend_from_slice(&digest_bytes);
        CurtaBytes(digest_bytes)
    }

    /*
    fn constrain_blake2b_gadget<C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone>(
        &mut self,
        gadget: Self::Gadget,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        // Allocate public input targets
        let public_blake2b_targets =
            BLAKE2BPublicData::add_virtual(self, &gadget.digest, &gadget.chunk_size);

        // Make the air
        let mut air_builder = AirBuilder::<BLAKE2BAirParameters<F, E>>::new();
        let clk = air_builder.clock();

        let (mut operations, table) = air_builder.byte_operations();

        let mut bus = air_builder.new_bus();
        let channel_idx = bus.new_channel(&mut air_builder);

        air_builder.blake2b_compress();

        let (air, trace_data) = air_builder.build();

        let generator = ArithmeticGenerator::<BLAKE2BAirParameters<F, E>>::new(trace_data);

        let public_input_target = public_blake2b_targets.public_input_targets(self);

        let blake_generator = BLAKE2BGenerator {
            gadget: blake_gadget,
            table,
            padded_message: gadget.padded_message,
            chunk_size: gadget.chunk_size,
            trace_generator: generator.clone(),
            pub_values_target: public_blake_target,
        };

        self.add_simple_generator(blake_generator);

        let stark = Starky::new(air);
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
    */
}

#[cfg(test)]
mod tests {

    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::hash::blake::blake2b::BLAKE2BGadget;

    #[test]
    fn test_blake_2b_plonky_gadget() {
        type F = GoldilocksField;
        type E = GoldilocksCubicParameters;
        type C = PoseidonGoldilocksConfig;
        const D: usize = 2;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Blake2b Plonky2 gadget test", log::Level::Debug);

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let mut gadget: BLAKE2BBuilderGadget<F, E, D> = builder.init_blake2b();

        let msg_target = CurtaBytes(builder.add_virtual_target_arr::<256>());
        let msg_length_target = builder.add_virtual_target();

        let calculated_digest = builder.blake2b(&msg_target, msg_length_target, &mut gadget);
        let expected_digest_target = CurtaBytes(builder.add_virtual_target_arr::<32>());

        for (d, e) in calculated_digest
            .0
            .iter()
            .zip(expected_digest_target.0.iter())
        {
            builder.connect(*d, *e);
        }

        //builder.constrain_blake2b_gadget::<C>(gadget);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();

        /*
        let msg = decode("").unwrap();
        let digest = "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8";

        let msg = b"abc".to_vec();
        let digest = "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319";

        let msg = b"243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917".to_vec();
        let digest = "486ce0fdbd0e2f6b798d1ef3d881585b7a3331802a995d4b7fdf886b8b03a9a4";
        */

        let msg = b"243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917".to_vec();
        let digest = "369ffcc61c51d8ed04bf30a9e8cf79f8994784d1e3f90f32c3182e67873a3238";

        let padded_msg = BLAKE2BGadget::pad(&msg)
            .into_iter()
            .map(F::from_canonical_u8)
            .collect::<Vec<_>>();

        let expected_digest = hex::decode(digest)
            .unwrap()
            .into_iter()
            .map(F::from_canonical_u8)
            .collect::<Vec<_>>();

        pw.set_target_arr(&msg_target.0, &padded_msg);
        pw.set_target(msg_length_target, F::from_canonical_usize(msg.len()));
        pw.set_target_arr(&expected_digest_target.0, &expected_digest);

        let proof = timed!(
            timing,
            "Generate proof",
            plonky2::plonk::prover::prove(&data.prover_only, &data.common, pw, &mut timing)
        )
        .unwrap();
        timing.print();
        data.verify(proof).unwrap();
    }
}
