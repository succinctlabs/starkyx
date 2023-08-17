//! SHA256 Gadget
//!
//! In this example we will see how to use Curta to batch the SHA256 processing phase. We will
//! hash padded messages of total length of 65Kb.
//!
//! To run, use:
//! ```
//! cargo run --release --example sha_256
//! ```
//!
//!  To see the timing print, use:
//! ```
//! RUST_LOG="debug" cargo run --release --example sha_256
//! ```
//!
//! For maximum performence, set the compiler flag to optimize for your CPU architecture:
//! ```
//! RUST_LOG="debug" RUSTFLAGS=-Ctarget-cpu=native cargo run --release --example sha_256
//! ```
//!

use curta::chip::hash::sha::sha256::builder_gadget::{
    CurtaBytes, SHA256Builder, SHA256BuilderGadget,
};
use curta::chip::hash::sha::sha256::SHA256Gadget;
use curta::math::goldilocks::cubic::GoldilocksCubicParameters;
use curta::math::prelude::*;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::plonk::prover::prove;
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use subtle_encoding::hex::decode;

fn main() {
    // Define some type aliases for convenience
    type F = GoldilocksField;
    type E = GoldilocksCubicParameters;
    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;

    // env-logger for timing information
    env_logger::init();

    let mut timing = TimingTree::new("Sha256 Plonky2 gadget", log::Level::Debug);

    // Create a new plonky2 circuit
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // Initialize the SHA gadget
    let mut gadget: SHA256BuilderGadget<F, E, D> = builder.init_sha_256();

    // We will hash 256 messages of padded length 128 bytes and 512 messages of padded length 64 bytes.

    // Initialize targets
    let long_padded_msg_targets = (0..256)
        .map(|_| CurtaBytes(builder.add_virtual_target_arr::<128>()))
        .collect::<Vec<_>>();

    let short_padded_msg_targets = (0..512)
        .map(|_| CurtaBytes(builder.add_virtual_target_arr::<64>()))
        .collect::<Vec<_>>();

    let mut digest_targets = Vec::new();
    let mut expected_digests = Vec::new();

    for long_padded_msg in long_padded_msg_targets.iter() {
        // Get the hash of the padded message
        let digest = builder.sha_256(long_padded_msg, &mut gadget);
        digest_targets.push(digest);
        let expected_digest = CurtaBytes(builder.add_virtual_target_arr::<32>());
        expected_digests.push(expected_digest);
    }

    for padded_msg in short_padded_msg_targets.iter() {
        // Get the hash of the padded message
        let digest = builder.sha_256(padded_msg, &mut gadget);
        digest_targets.push(digest);
        let expected_digest = CurtaBytes(builder.add_virtual_target_arr::<32>());
        expected_digests.push(expected_digest);
    }

    // Connect the expected and output digests
    for (digest, expected) in digest_targets.iter().zip(expected_digests.iter()) {
        for (d, e) in digest.0.iter().zip(expected.0.iter()) {
            builder.connect(*d, *e);
        }
    }

    // Register the SHA constraints in the builder
    builder.constrain_sha_256_gadget::<C>(gadget);

    // Build the circuit
    let data = builder.build::<C>();

    // Assign input values and make the proof
    let mut pw = PartialWitness::new();

    // We will use two types of a short message and one long message

    let short_msg_1 = decode("").unwrap();
    let expected_digest_1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    let short_msg_2 = b"abc".to_vec();
    let expected_digest_2 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    let long_msg = decode("243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917").unwrap();
    let expected_digest_long = "aca16131a2e4c4c49e656d35aac1f0e689b3151bb108fa6cf5bcc3ac08a09bf9";

    let long_messages = (0..256).map(|_| long_msg.clone()).collect::<Vec<_>>();

    // Pad the long messages
    let padded_long_messages = long_messages
        .iter()
        .map(|m| {
            SHA256Gadget::pad(m)
                .into_iter()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Pad the short messages
    let short_messages = (0..256)
        .flat_map(|_| [short_msg_1.clone(), short_msg_2.clone()])
        .collect::<Vec<_>>();
    assert_eq!(short_messages.len(), 512);
    let padded_short_messages = short_messages
        .iter()
        .map(|m| {
            SHA256Gadget::pad(m)
                .into_iter()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Get the expected digest values
    let expected_digests_long_message = (0..256)
        .map(|_| expected_digest_long)
        .map(|digest| {
            hex::decode(digest)
                .unwrap()
                .into_iter()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let expected_digests_short_messages = (0..256)
        .flat_map(|_| [expected_digest_1, expected_digest_2])
        .map(|digest| {
            hex::decode(digest)
                .unwrap()
                .into_iter()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut expected_digests_values = expected_digests_long_message;
    expected_digests_values.extend(expected_digests_short_messages);

    // Assign the inputs
    for (msg_target, long_msg) in long_padded_msg_targets
        .iter()
        .zip(padded_long_messages.iter())
    {
        pw.set_target_arr(&msg_target.0, long_msg);
    }

    for (msg_target, short_msg) in short_padded_msg_targets
        .iter()
        .zip(padded_short_messages.iter())
    {
        pw.set_target_arr(&msg_target.0, short_msg);
    }

    for (digest, value) in expected_digests.iter().zip(expected_digests_values.iter()) {
        pw.set_target_arr(&digest.0, value);
    }

    // Generate the proof
    let proof = timed!(
        timing,
        "Generate proof",
        prove(&data.prover_only, &data.common, pw, &mut timing)
    )
    .unwrap();
    timing.print();

    // Verify the proof
    data.verify(proof).unwrap();
}
