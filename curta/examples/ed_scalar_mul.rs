//! Plonky2 EdDSA Signature Gadget
//!
//! In this example we will see how to use Curta to verify a batch of 256 EdDSA signatures.
//!
//! To run, use:
//! ```
//! cargo run --release --example ed_scalar_mul
//! ```
//!
//!  To see the timing print, use:
//! ```
//! RUST_LOG="debug" cargo run --release --example ed_scalar_mul
//! ```
//!
//! For maximum performence, set the compiler flag to optimize for your CPU architecture:
//! ```
//! RUST_LOG="debug" RUSTFLAGS=-Ctarget-cpu=native cargo run --release --example ed_scalar_mul
//! ```

use curta::chip::ec::edwards::ed25519::Ed25519;
use curta::chip::ec::edwards::scalar_mul::generator::{
    AffinePointTarget, EdDSAStark, ScalarMulEd25519Gadget,
};
use curta::chip::ec::edwards::EdwardsParameters;
use curta::chip::utils::biguint_to_16_digits_field;
use curta::math::goldilocks::cubic::GoldilocksCubicParameters;
use num::bigint::RandBigInt;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use rand::thread_rng;

fn main() {
    // Devlarying some type aliases for convenience
    type F = GoldilocksField;
    type E = GoldilocksCubicParameters;
    type C = PoseidonGoldilocksConfig;
    type S = EdDSAStark<F, E>;
    const D: usize = 2;

    // env-logger for timing information
    env_logger::init();

    // Create a new plonky2 circuit
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let mut timing = TimingTree::new("Ed25519 Scalar mul", log::Level::Debug);

    // Allocate targets for elliptic curve points and scalar

    // Get virtual targets for scalars
    let scalars_limbs = (0..256)
        .map(|_| {
            (0..8)
                .map(|_| builder.add_virtual_target())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for scalar in scalars_limbs.iter() {
        for limb in scalar.iter() {
            builder.register_public_input(*limb);
        }
    }

    // Get virtual targets for points
    let points = (0..256)
        .map(|_| {
            let x = builder.add_virtual_target_arr();
            let y = builder.add_virtual_target_arr();
            AffinePointTarget { x, y }
        })
        .collect::<Vec<_>>();

    for point in points.iter() {
        builder.register_public_inputs(&point.x);
        builder.register_public_inputs(&point.y);
    }

    // Get the results of the scalar multiplication
    let results = builder.ed_scalar_mul_batch::<S, E, C>(&points, &scalars_limbs);

    // These results will be allocated automatically into the trace once `points` and
    // `scalars` are written into the trace.
    // We can compare these results with the expected results by writing these explicitely
    // into the trace.
    let expected_results = (0..256)
        .map(|_| {
            let x = builder.add_virtual_target_arr();
            let y = builder.add_virtual_target_arr();
            AffinePointTarget { x, y }
        })
        .collect::<Vec<_>>();

    for point in expected_results.iter() {
        builder.register_public_inputs(&point.x);
        builder.register_public_inputs(&point.y);
    }

    // Compare the results to the expeced results
    for (res, expected) in results.iter().zip(expected_results.iter()) {
        builder.connect_affine_point(res, expected);
    }

    // Build the circuit
    let data = builder.build::<C>();

    // We can now assign the public inputs and make the proof
    let mut pw = PartialWitness::new();

    // Assigning the public inputs: points, scalars, and expected results
    let mut rng = thread_rng();
    let generator = Ed25519::generator();
    timed!(
        timing,
        "assigning inputs",
        for i in 0..256 {
            let a = rng.gen_biguint(256);
            let point = &generator * a;
            let scalar = rng.gen_biguint(256);
            let res = &point * &scalar;

            //Set the expected result
            let res_limbs_x: [_; 16] = biguint_to_16_digits_field(&res.x, 16).try_into().unwrap();
            let res_limbs_y: [_; 16] = biguint_to_16_digits_field(&res.y, 16).try_into().unwrap();
            pw.set_target_arr(&expected_results[i].x, &res_limbs_x);
            pw.set_target_arr(&expected_results[i].y, &res_limbs_y);

            // Set the scalar target
            let mut scalar_limbs = scalar
                .iter_u32_digits()
                .map(F::from_canonical_u32)
                .collect::<Vec<_>>();
            scalar_limbs.resize(8, F::ZERO);
            for (target, limb) in scalars_limbs[i].iter().zip(scalar_limbs) {
                pw.set_target(*target, limb);
            }

            // Set the point target
            let point_limbs_x: [_; 16] =
                biguint_to_16_digits_field(&point.x, 16).try_into().unwrap();
            let point_limbs_y: [_; 16] =
                biguint_to_16_digits_field(&point.y, 16).try_into().unwrap();

            pw.set_target_arr(&points[i].x, &point_limbs_x);
            pw.set_target_arr(&points[i].y, &point_limbs_y);
        }
    );

    let recursive_proof = timed!(
        timing,
        "Generate proof",
        plonky2::plonk::prover::prove(&data.prover_only, &data.common, pw, &mut timing)
    )
    .unwrap();
    timing.print();
    data.verify(recursive_proof).unwrap();
}
