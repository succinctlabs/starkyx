use core::fmt::Debug;
use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use serde::{Deserialize, Serialize};

use super::generator::{BLAKE2BGenerator, BLAKE2BHintGenerator, BLAKE2BStarkData};
use super::{BLAKE2BPublicData, NUM_MIX_ROUNDS};
use crate::chip::hash::CurtaBytes;
use crate::chip::AirParameters;
use crate::math::prelude::CubicParameters;
use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::gadget::StarkGadget;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BLAKE2BBuilderGadget<
    F,
    E,
    const D: usize,
    L: AirParameters + 'static + Clone + Debug + Send + Sync,
> {
    pub padded_messages: Vec<Target>,
    pub msg_lengths: Vec<Target>,
    pub digests: Vec<Target>,
    pub _marker: PhantomData<(F, E, L)>,
}

pub trait BLAKE2BBuilder<
    F: RichField + Extendable<D>,
    E: CubicParameters<F>,
    const D: usize,
    L: AirParameters + 'static + Clone + Debug + Send + Sync,
>
{
    type Gadget;

    fn init_blake2b(&mut self) -> Self::Gadget;

    fn blake2b<const N: usize>(
        &mut self,
        padded_message: &CurtaBytes<N>,
        message_len: Target,
        gadget: &mut Self::Gadget,
    ) -> CurtaBytes<32>;

    fn constrain_blake2b_gadget<C: CurtaConfig<D, F = F, FE = F::Extension>>(
        &mut self,
        gadget: Self::Gadget,
    );

    fn max_num_chunks() -> usize;
}

impl<
        F: RichField + Extendable<D>,
        E: CubicParameters<F>,
        const D: usize,
        L: AirParameters + 'static + Clone + Debug + Send + Sync,
    > BLAKE2BBuilder<F, E, D, L> for CircuitBuilder<F, D>
{
    type Gadget = BLAKE2BBuilderGadget<F, E, D, L>;

    fn init_blake2b(&mut self) -> Self::Gadget {
        BLAKE2BBuilderGadget {
            padded_messages: Vec::new(),
            msg_lengths: Vec::new(),
            digests: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn blake2b<const N: usize>(
        &mut self,
        padded_message: &CurtaBytes<N>,
        message_len: Target,
        gadget: &mut Self::Gadget,
    ) -> CurtaBytes<32> {
        gadget.padded_messages.extend_from_slice(&padded_message.0);
        let digest_bytes = self.add_virtual_target_arr::<32>();
        let hint = BLAKE2BHintGenerator::new(&padded_message.0, message_len, digest_bytes);
        self.add_simple_generator(hint);
        gadget.digests.extend_from_slice(&digest_bytes);
        gadget.msg_lengths.push(message_len);
        CurtaBytes(digest_bytes)
    }

    fn constrain_blake2b_gadget<C: CurtaConfig<D, F = F, FE = F::Extension>>(
        &mut self,
        gadget: Self::Gadget,
    ) {
        // Allocate public input targets
        let public_blake2b_targets = BLAKE2BPublicData::add_virtual::<F, D, L>(self);

        let stark_data = BLAKE2BGenerator::<F, E, C, D, L>::stark_data();
        let BLAKE2BStarkData { stark, config, .. } = stark_data;

        let public_input_target = public_blake2b_targets.public_input_targets(self);

        let virtual_proof = self.add_virtual_stark_proof(&stark, &config);
        self.verify_stark_proof(&config, &stark, &virtual_proof, &public_input_target);

        let blake2b_generator = BLAKE2BGenerator::<F, E, C, D, L> {
            padded_messages: gadget.padded_messages,
            msg_lens: gadget.msg_lengths,
            pub_values_target: public_blake2b_targets,
            config,
            proof_target: virtual_proof,
            _phantom: PhantomData,
        };

        self.add_simple_generator(blake2b_generator);
    }

    fn max_num_chunks() -> usize {
        L::num_rows() / NUM_MIX_ROUNDS
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

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::hash::blake::blake2b::generator::BLAKE2BAirParameters;
    use crate::chip::hash::blake::blake2b::BLAKE2BGadget;
    use crate::plonky2::stark::config::CurtaPoseidonGoldilocksConfig;

    #[test]
    fn test_blake_2b_plonky_gadget() {
        type F = GoldilocksField;
        type E = GoldilocksCubicParameters;
        type SC = CurtaPoseidonGoldilocksConfig;
        type C = PoseidonGoldilocksConfig;
        type L = BLAKE2BAirParameters<F, E>;
        const D: usize = 2;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Blake2b Plonky2 gadget test", log::Level::Debug);

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let mut gadget: BLAKE2BBuilderGadget<F, E, D, L> = builder.init_blake2b();

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

        builder.constrain_blake2b_gadget::<SC>(gadget);

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
