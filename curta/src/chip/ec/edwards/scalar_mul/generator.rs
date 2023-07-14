use core::fmt::Debug;
use std::sync::mpsc::channel;

use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::packable::Packable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use super::air::ScalarMulEd25519;
use super::gadget::EdScalarMulGadget;
use crate::air::RAir;
use crate::chip::ec::edwards::ed25519::Ed25519;
use crate::chip::ec::point::AffinePoint;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::utils::{biguint_to_16_digits_field, field_limbs_to_biguint};
use crate::chip::{AirParameters, Chip};
use crate::maybe_rayon::*;
use crate::plonky2::field::CubicParameters;
use crate::plonky2::parser::{RecursiveStarkParser, StarkParser};
use crate::plonky2::stark::config::StarkyConfig;
use crate::plonky2::stark::gadget::StarkGadget;
use crate::plonky2::stark::generator::simple::SimpleStarkWitnessGenerator;
use crate::plonky2::stark::{Plonky2Stark, Starky};
use crate::trace::generator::TraceGenerator;

pub type EdDSAStark<F, E> = Starky<Chip<ScalarMulEd25519<F, E>>, { 2330 + 1504 }>;

#[derive(Debug, Clone, Copy)]
pub struct AffinePointTarget {
    pub x: [Target; 16],
    pub y: [Target; 16],
}

pub trait ScalarMulEd25519Gadget<F: RichField + Extendable<D>, const D: usize> {
    fn ed_scalar_mul_batch<
        S: Plonky2Stark<F, D> + 'static + Send + Sync + Debug,
        E: CubicParameters<F>,
        C: GenericConfig<D, F = F, FE = F::Extension> + 'static,
    >(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<BoolTarget>],
    ) -> Vec<AffinePointTarget>
    where
        C::Hasher: AlgebraicHasher<F>,
        S::Air: for<'a> RAir<RecursiveStarkParser<'a, F, D>>
            + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>,
        ArithmeticGenerator<ScalarMulEd25519<F, E>>: TraceGenerator<F, S::Air>,
        <ArithmeticGenerator<ScalarMulEd25519<F, E>> as TraceGenerator<F, S::Air>>::Error:
            Into<anyhow::Error>,
        S: From<Starky<Chip<ScalarMulEd25519<F, E>>, { 2330 + 1504 }>>,
        [(); S::COLUMNS]:;
}

impl<F: RichField + Extendable<D>, const D: usize> ScalarMulEd25519Gadget<F, D>
    for CircuitBuilder<F, D>
{
    fn ed_scalar_mul_batch<
        S: Plonky2Stark<F, D> + 'static + Send + Sync + Debug,
        E: CubicParameters<F>,
        C: GenericConfig<D, F = F, FE = F::Extension> + 'static,
    >(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<BoolTarget>],
    ) -> Vec<AffinePointTarget>
    where
        C::Hasher: AlgebraicHasher<F>,
        S::Air: for<'a> RAir<RecursiveStarkParser<'a, F, D>>
            + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>,
        ArithmeticGenerator<ScalarMulEd25519<F, E>>: TraceGenerator<F, S::Air>,
        <ArithmeticGenerator<ScalarMulEd25519<F, E>> as TraceGenerator<F, S::Air>>::Error:
            Into<anyhow::Error>,
        S: From<Starky<Chip<ScalarMulEd25519<F, E>>, { 2330 + 1504 }>>,
        [(); S::COLUMNS]:,
    {
        let (air, gadget) = ScalarMulEd25519::<F, E>::air();

        let stark = Starky::<_, { 2330 + 1504 }>::new(air); //TODO: MAKE SURE NUM_COLS FITS
        let config =
            StarkyConfig::<F, C, D>::standard_fast_config(ScalarMulEd25519::<F, E>::num_rows());
        let virtual_proof = self.add_virtual_stark_proof(&stark, &config);
        self.verify_stark_proof(&config, &stark, virtual_proof.clone(), &[]);

        let stark_generator = SimpleStarkWitnessGenerator::new(
            config,
            stark.into(),
            virtual_proof,
            vec![],
            ArithmeticGenerator::<ScalarMulEd25519<F, E>>::new(&[]),
        );

        let results = (0..256)
            .into_iter()
            .map(|_| {
                let x = self.add_virtual_target_arr();
                let y = self.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        let generator = SimpleScalarMulEd25519Generator {
            gadget,
            points: points.to_vec(),
            scalars: scalars.to_vec(),
            generator: stark_generator,
            results: results.clone(),
            _marker: core::marker::PhantomData,
        };

        self.add_simple_generator(generator);
        results
    }
}

#[derive(Debug, Clone)]
pub struct SimpleScalarMulEd25519Generator<
    F: RichField + Extendable<D>,
    E: CubicParameters<F>,
    C: GenericConfig<D, F = F>,
    S,
    const D: usize,
> {
    gadget: EdScalarMulGadget<Ed25519>,
    points: Vec<AffinePointTarget>,
    scalars: Vec<Vec<BoolTarget>>, // 32-byte limbs
    results: Vec<AffinePointTarget>,
    generator: SimpleStarkWitnessGenerator<
        S,
        ArithmeticGenerator<ScalarMulEd25519<F, E>>,
        F,
        C,
        <F as Packable>::Packing,
        D,
    >,
    _marker: core::marker::PhantomData<(F, C, E)>,
}

impl<
        F: RichField + Extendable<D>,
        E: CubicParameters<F>,
        C: GenericConfig<D, F = F>,
        S,
        const D: usize,
    > SimpleScalarMulEd25519Generator<F, E, C, S, D>
{
    pub fn new(
        gadget: EdScalarMulGadget<Ed25519>,
        points: Vec<AffinePointTarget>,
        scalars: Vec<Vec<BoolTarget>>,
        results: Vec<AffinePointTarget>,
        generator: SimpleStarkWitnessGenerator<
            S,
            ArithmeticGenerator<ScalarMulEd25519<F, E>>,
            F,
            C,
            <F as Packable>::Packing,
            D,
        >,
    ) -> Self {
        Self {
            gadget,
            points,
            scalars,
            results,
            generator,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<
        F: RichField + Extendable<D>,
        E: CubicParameters<F>,
        C: GenericConfig<D, F = F, FE = F::Extension> + 'static,
        S: Plonky2Stark<F, D> + 'static + Send + Sync + Debug,
        const D: usize,
    > SimpleGenerator<F> for SimpleScalarMulEd25519Generator<F, E, C, S, D>
where
    C::Hasher: AlgebraicHasher<F>,
    S::Air: for<'a> RAir<RecursiveStarkParser<'a, F, D>>
        + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>,
    ArithmeticGenerator<ScalarMulEd25519<F, E>>: TraceGenerator<F, S::Air>,
    <ArithmeticGenerator<ScalarMulEd25519<F, E>> as TraceGenerator<F, S::Air>>::Error:
        Into<anyhow::Error>,
    [(); S::COLUMNS]:,
{
    fn dependencies(&self) -> Vec<Target> {
        self.points
            .iter()
            .flat_map(|point| [point.x.to_vec(), point.y.to_vec()].into_iter().flatten())
            .chain(
                self.scalars
                    .iter()
                    .flat_map(|scalar| scalar.into_iter().map(|x| x.target)),
            )
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let scalars = self
            .scalars
            .iter()
            .map(|x| {
                x.iter()
                    .map(|bool| witness.get_target(bool.target))
                    .collect::<Vec<_>>()
            })
            .map(|bits| {
                let mut x_out = BigUint::from(0u32);
                for (i, bit) in bits.into_iter().enumerate() {
                    if bit == F::ONE {
                        x_out += BigUint::from(1u32) << i;
                    }
                }
                x_out
            })
            .collect::<Vec<_>>();
        assert_eq!(scalars.len(), 256);

        let points = self
            .points
            .iter()
            .map(|point| {
                let x = field_limbs_to_biguint(&witness.get_targets(&point.x));
                let y = field_limbs_to_biguint(&witness.get_targets(&point.y));
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        assert_eq!(points.len(), 256);

        // Generate the trace
        let trace_generator = &self.generator.trace_generator;

        let mut counter_val = F::ONE;
        let counter_gen = F::primitive_root_of_unity(8);
        let writer = trace_generator.new_writer();
        for j in 0..(1 << 16) {
            writer.write_value(&self.gadget.cyclic_counter, &counter_val, j);
            counter_val *= counter_gen;
        }
        let (tx, rx) = channel();
        for i in 0..256usize {
            let writer = trace_generator.new_writer();
            let tx = tx.clone();
            let gadget = self.gadget.clone();
            let point = points[i].clone();
            let scalar = scalars[i].clone();
            rayon::spawn(move || {
                let res = writer.write_ed_double_and_add(
                    &scalar,
                    &point,
                    &gadget.double_and_add_gadget,
                    256 * i,
                );
                tx.send((i, res)).unwrap();
            });
        }
        drop(tx);
        for (i, res) in rx.iter() {
            let res_limbs_x: [_; 16] = biguint_to_16_digits_field(&res.x, 16).try_into().unwrap();
            let res_limbs_y: [_; 16] = biguint_to_16_digits_field(&res.y, 16).try_into().unwrap();
            out_buffer.set_target_arr(self.results[i].x, res_limbs_x);
            out_buffer.set_target_arr(self.results[i].y, res_limbs_y);
        }

        // Generate the stark proof
        SimpleGenerator::<F>::run_once(&self.generator, witness, out_buffer)
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use rand::thread_rng;

    use super::*;
    use crate::chip::ec::edwards::scalar_mul::generator::{
        AffinePointTarget, ScalarMulEd25519Gadget,
    };
    use crate::chip::ec::edwards::EdwardsParameters;
    use crate::chip::utils::biguint_to_bits_le;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

    #[test]
    fn test_scalar_gadget() {
        type F = GoldilocksField;
        type E = GoldilocksCubicParameters;
        type C = PoseidonGoldilocksConfig;
        type S = EdDSAStark<F, E>;
        const D: usize = 2;

        let _ = env_logger::builder().is_test(true).try_init();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // Get virtual targets for scalars,  points
        let scalars = (0..256)
            .into_iter()
            .map(|_| {
                (0..256)
                    .into_iter()
                    .map(|_| builder.add_virtual_bool_target_unsafe())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for scalar in scalars.iter() {
            for bool in scalar.iter() {
                builder.register_public_input(bool.target);
            }
        }

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

        // The scalar multiplications
        let results = builder.ed_scalar_mul_batch::<S, E, C>(&points, &scalars);

        // Get virtual targets for scalars,  points
        let scalars_2 = (0..256)
            .into_iter()
            .map(|_| {
                (0..256)
                    .into_iter()
                    .map(|_| builder.add_virtual_bool_target_unsafe())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for scalar in scalars.iter() {
            for bool in scalar.iter() {
                builder.register_public_input(bool.target);
            }
        }

        let points_2 = (0..256)
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
        let results_2 = builder.ed_scalar_mul_batch::<S, E, C>(&points, &scalars);

        // Compare the results to the expeced results
        for (res, expected) in results.iter().zip(expected_results.iter()) {
            for k in 0..16 {
                builder.connect(res.x[k], expected.x[k]);
                builder.connect(res.y[k], expected.y[k]);
            }
        }

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();

        let mut rng = thread_rng();
        let generator = Ed25519::generator();
        let nb_bits = Ed25519::nb_scalar_bits();

        for i in 0..256 {
            let a = rng.gen_biguint(256);
            let b = rng.gen_biguint(256);
            let point = &generator * a;
            let point_b = &generator * b;
            let scalar = rng.gen_biguint(256);
            let scalar_2 = rng.gen_biguint(256);
            let res = &point * &scalar;

            //Set the expected result
            let res_limbs_x: [_; 16] = biguint_to_16_digits_field(&res.x, 16).try_into().unwrap();
            let res_limbs_y: [_; 16] = biguint_to_16_digits_field(&res.y, 16).try_into().unwrap();
            pw.set_target_arr(expected_results[i].x, res_limbs_x);
            pw.set_target_arr(expected_results[i].y, res_limbs_y);

            // Set the scalar target
            let scalar_bits = biguint_to_bits_le(&scalar, nb_bits);
            for (target, bit) in scalars[i].iter().zip(scalar_bits.iter()) {
                pw.set_bool_target(*target, *bit);
            }

            // Set the point target
            let point_limbs_x: [_; 16] =
                biguint_to_16_digits_field(&point.x, 16).try_into().unwrap();
            let point_limbs_y: [_; 16] =
                biguint_to_16_digits_field(&point.y, 16).try_into().unwrap();

            pw.set_target_arr(points[i].x, point_limbs_x);
            pw.set_target_arr(points[i].y, point_limbs_y);

            // Set the scalar target
            let scalar_bits = biguint_to_bits_le(&scalar_2, nb_bits);
            for (target, bit) in scalars_2[i].iter().zip(scalar_bits.iter()) {
                pw.set_bool_target(*target, *bit);
            }

            // Set the point target
            let point_limbs_2_x: [_; 16] = biguint_to_16_digits_field(&point_b.x, 16)
                .try_into()
                .unwrap();
            let point_limbs_2_y: [_; 16] = biguint_to_16_digits_field(&point_b.y, 16)
                .try_into()
                .unwrap();

            pw.set_target_arr(points_2[i].x, point_limbs_2_x);
            pw.set_target_arr(points_2[i].y, point_limbs_2_y);
        }

        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
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
