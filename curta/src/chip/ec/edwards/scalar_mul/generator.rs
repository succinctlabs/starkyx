use core::fmt::Debug;
use std::sync::mpsc::channel;

use itertools::Itertools;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::packable::Packable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use super::air::{ScalarMulEd25519, ED_NUM_COLUMNS};
use super::gadget::EdScalarMulGadget;
use crate::air::RAir;
use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField};
use crate::chip::ec::point::AffinePoint;
use crate::chip::ec::EllipticCurveParameters;
use crate::chip::field::instruction::FpInstruction;
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::RegisterSerializable;
use crate::chip::table::evaluation::BitEvaluation;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::utils::{biguint_to_16_digits_field, field_limbs_to_biguint};
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::CubicParameters;
use crate::plonky2::parser::{RecursiveStarkParser, StarkParser};
use crate::plonky2::stark::config::StarkyConfig;
use crate::plonky2::stark::gadget::StarkGadget;
use crate::plonky2::stark::generator::simple::SimpleStarkWitnessGenerator;
use crate::plonky2::stark::{Plonky2Stark, Starky};
use crate::trace::generator::TraceGenerator;

pub type EdDSAStark<F, E> = Starky<Chip<ScalarMulEd25519<F, E>>, ED_NUM_COLUMNS>;

const AFFINE_POINT_TARGET_NUM_LIMBS: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct AffinePointTarget {
    pub x: [Target; AFFINE_POINT_TARGET_NUM_LIMBS],
    pub y: [Target; AFFINE_POINT_TARGET_NUM_LIMBS],
}

pub trait ScalarMulEd25519Gadget<F: RichField + Extendable<D>, const D: usize> {
    fn ed_scalar_mul_batch<
        S: Plonky2Stark<F, D> + 'static + Send + Sync + Debug + Clone,
        E: CubicParameters<F>,
        C: GenericConfig<D, F = F, FE = F::Extension> + 'static,
    >(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<Target>],
    ) -> Vec<AffinePointTarget>
    where
        C::Hasher: AlgebraicHasher<F>,
        S::Air: for<'a> RAir<RecursiveStarkParser<'a, F, D>>
            + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>,
        ArithmeticGenerator<ScalarMulEd25519<F, E>>: TraceGenerator<F, S::Air>,
        <ArithmeticGenerator<ScalarMulEd25519<F, E>> as TraceGenerator<F, S::Air>>::Error:
            Into<anyhow::Error>,
        S: From<Starky<Chip<ScalarMulEd25519<F, E>>, ED_NUM_COLUMNS>>,
        [(); S::COLUMNS]:;

    fn ed_scalar_mul_batch_hint(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<Target>],
    ) -> Vec<AffinePointTarget>;

    fn connect_affine_point(&mut self, lhs: &AffinePointTarget, rhs: &AffinePointTarget);

    fn constant_affine_point<EP: EllipticCurveParameters>(
        &mut self,
        point: AffinePoint<EP>,
    ) -> AffinePointTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> ScalarMulEd25519Gadget<F, D>
    for CircuitBuilder<F, D>
{
    fn ed_scalar_mul_batch<
        S: Plonky2Stark<F, D> + 'static + Send + Sync + Debug + Clone,
        E: CubicParameters<F>,
        C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone,
    >(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<Target>],
    ) -> Vec<AffinePointTarget>
    where
        C::Hasher: AlgebraicHasher<F>,
        S::Air: for<'a> RAir<RecursiveStarkParser<'a, F, D>>
            + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>,
        ArithmeticGenerator<ScalarMulEd25519<F, E>>: TraceGenerator<F, S::Air> + Clone,
        <ArithmeticGenerator<ScalarMulEd25519<F, E>> as TraceGenerator<F, S::Air>>::Error:
            Into<anyhow::Error>,
        S: From<Starky<Chip<ScalarMulEd25519<F, E>>, ED_NUM_COLUMNS>>,
        [(); S::COLUMNS]:,
    {
        let (
            air,
            gadget,
            scalars_limbs_input,
            input_points,
            output_points,
            (bit_eval, set_last, set_bit),
        ) = ScalarMulEd25519::<F, E>::air();

        let mut public_input_target_option = vec![None as Option<Target>; 256 * (8 + 2 * 32)];
        for (scalar_register, scalar_target) in scalars_limbs_input.iter().zip_eq(scalars.iter()) {
            let (s_0, s_1) = scalar_register.register().get_range();
            let scalar_targets = scalar_target.iter().map(|x| Some(*x)).collect_vec();
            public_input_target_option[s_0..s_1].copy_from_slice(&scalar_targets);
        }

        for (point_register, point_target) in input_points.iter().zip(points.iter()) {
            let (p_x_0, p_x_1) = point_register.x.register().get_range();
            let (p_y_0, p_y_1) = point_register.y.register().get_range();
            assert_eq!(p_x_1, p_y_0, "x and y registers must be consecutive");
            let point_targets = point_target
                .x
                .iter()
                .chain(point_target.y.iter())
                .map(|v| Some(*v))
                .collect_vec();
            public_input_target_option[p_x_0..p_y_1].copy_from_slice(&point_targets);
        }

        // Input results
        let results = (0..256)
            .map(|_| {
                let x = self.add_virtual_target_arr();
                let y = self.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        for (point_register, point_target) in output_points.iter().zip(results.iter()) {
            let (p_x_0, p_x_1) = point_register.x.register().get_range();
            let (p_y_0, p_y_1) = point_register.y.register().get_range();
            assert_eq!(p_x_1, p_y_0, "x and y registers must be consecutive");
            let point_targets = point_target
                .x
                .iter()
                .chain(point_target.y.iter())
                .map(|v| Some(*v))
                .collect_vec();
            public_input_target_option[p_x_0..p_y_1].copy_from_slice(&point_targets);
        }

        let public_input_target = public_input_target_option
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec();

        let stark = Starky::<_, ED_NUM_COLUMNS>::new(air); //TODO: MAKE SURE NUM_COLS FITS
        let config =
            StarkyConfig::<F, C, D>::standard_fast_config(ScalarMulEd25519::<F, E>::num_rows());
        let virtual_proof = self.add_virtual_stark_proof(&stark, &config);
        self.verify_stark_proof(&config, &stark, virtual_proof.clone(), &public_input_target);

        let stark_generator = SimpleStarkWitnessGenerator::new(
            config,
            stark.into(),
            virtual_proof,
            public_input_target,
            ArithmeticGenerator::<ScalarMulEd25519<F, E>>::new(&[]),
        );

        let generator = SimpleScalarMulEd25519Generator {
            gadget,
            points: points.to_vec(),
            scalars: scalars.to_vec(),
            generator: stark_generator.clone(),
            results: results.clone(),
            bit_eval,
            set_last,
            set_bit,
            _marker: core::marker::PhantomData,
        };

        self.add_simple_generator(generator);
        self.add_simple_generator(stark_generator);
        results
    }

    fn ed_scalar_mul_batch_hint(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<Target>],
    ) -> Vec<AffinePointTarget> {
        let results = (0..256)
            .map(|_| {
                let x = self.add_virtual_target_arr();
                let y = self.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        let generator = SimpleScalarMulEd25519HintGenerator::<F, D> {
            points: points.to_vec(),
            scalars: scalars.to_vec(),
            results: results.clone(),
            _marker: core::marker::PhantomData,
        };

        self.add_simple_generator(generator);
        results
    }

    fn connect_affine_point(&mut self, lhs: &AffinePointTarget, rhs: &AffinePointTarget) {
        for i in 0..AFFINE_POINT_TARGET_NUM_LIMBS {
            self.connect(lhs.x[i], rhs.x[i]);
            self.connect(lhs.y[i], rhs.y[i]);
        }
    }

    fn constant_affine_point<EP: EllipticCurveParameters>(
        &mut self,
        point: AffinePoint<EP>,
    ) -> AffinePointTarget {
        let x_limbs: [_; AFFINE_POINT_TARGET_NUM_LIMBS] =
            biguint_to_16_digits_field::<F>(&point.x, 16)
                .iter()
                .map(|x| self.constant(*x))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        let y_limbs: [_; AFFINE_POINT_TARGET_NUM_LIMBS] =
            biguint_to_16_digits_field::<F>(&point.y, 16)
                .iter()
                .map(|x| self.constant(*x))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

        AffinePointTarget {
            x: x_limbs,
            y: y_limbs,
        }
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
    gadget: EdScalarMulGadget<F, Ed25519>,
    points: Vec<AffinePointTarget>,
    scalars: Vec<Vec<Target>>, // 32-byte limbs
    results: Vec<AffinePointTarget>,
    bit_eval: BitEvaluation<F>,
    set_last: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
    set_bit: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gadget: EdScalarMulGadget<F, Ed25519>,
        points: Vec<AffinePointTarget>,
        scalars: Vec<Vec<Target>>,
        results: Vec<AffinePointTarget>,
        bit_eval: BitEvaluation<F>,
        set_last: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
        set_bit: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
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
            bit_eval,
            set_last,
            set_bit,
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
    > SimpleGenerator<F, D> for SimpleScalarMulEd25519Generator<F, E, C, S, D>
where
    C::Hasher: AlgebraicHasher<F>,
    S::Air: for<'a> RAir<RecursiveStarkParser<'a, F, D>>
        + for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>,
    ArithmeticGenerator<ScalarMulEd25519<F, E>>: TraceGenerator<F, S::Air>,
    <ArithmeticGenerator<ScalarMulEd25519<F, E>> as TraceGenerator<F, S::Air>>::Error:
        Into<anyhow::Error>,
    [(); S::COLUMNS]:,
{
    fn id(&self) -> String {
        unimplemented!("TODO")
    }

    fn dependencies(&self) -> Vec<Target> {
        self.points
            .iter()
            .flat_map(|point| [point.x.to_vec(), point.y.to_vec()].into_iter().flatten())
            .chain(
                self.scalars
                    .iter()
                    .flat_map(|scalar| scalar.iter().copied()),
            )
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let scalars = self
            .scalars
            .par_iter()
            .map(|x| {
                x.iter()
                    .map(|limb| witness.get_target(*limb))
                    .map(|x| F::as_canonical_u64(&x) as u32)
                    .collect::<Vec<_>>()
            })
            .map(BigUint::new)
            .collect::<Vec<_>>();
        assert_eq!(scalars.len(), 256);

        let points = self
            .points
            .par_iter()
            .map(|point| {
                let x = field_limbs_to_biguint(&witness.get_targets(&point.x));
                let y = field_limbs_to_biguint(&witness.get_targets(&point.y));
                AffinePoint::new(x, y)
            })
            .collect::<Vec<_>>();
        assert_eq!(points.len(), 256);

        // Generate the trace
        let trace_generator = &self.generator.trace_generator;

        let writer = trace_generator.new_writer();
        for j in 0..(1 << 16) {
            writer.write_instruction(&self.gadget.cycle, j);
            writer.write_instruction(&self.bit_eval.cycle, j);
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
            out_buffer.set_target_arr(&self.results[i].x, &res_limbs_x);
            out_buffer.set_target_arr(&self.results[i].y, &res_limbs_y);
        }
        for j in (0..(1 << 16)).rev() {
            writer.write_instruction(&self.set_last, j);
            writer.write_instruction(&self.set_bit, j);
        }
        // Generate the stark proof
        // SimpleGenerator::<F, D>::run_once(&self.generator, witness, out_buffer)
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        unimplemented!("SimpleScalarMulEd25519Generator::serialize")
    }

    fn deserialize(
        _src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self> {
        unimplemented!("SimpleScalarMulEd25519Generator::deserialize")
    }
}

#[derive(Debug, Clone)]
pub struct SimpleScalarMulEd25519HintGenerator<F: RichField + Extendable<D>, const D: usize> {
    points: Vec<AffinePointTarget>,
    scalars: Vec<Vec<Target>>, // 32-byte limbs
    results: Vec<AffinePointTarget>,
    _marker: core::marker::PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleScalarMulEd25519HintGenerator<F, D> {
    pub fn new(
        points: Vec<AffinePointTarget>,
        scalars: Vec<Vec<Target>>,
        results: Vec<AffinePointTarget>,
    ) -> Self {
        Self {
            points,
            scalars,
            results,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for SimpleScalarMulEd25519HintGenerator<F, D>
{
    fn id(&self) -> String {
        unimplemented!("TODO")
    }

    fn dependencies(&self) -> Vec<Target> {
        self.points
            .iter()
            .flat_map(|point| [point.x.to_vec(), point.y.to_vec()].into_iter().flatten())
            .chain(
                self.scalars
                    .iter()
                    .flat_map(|scalar| scalar.iter().copied()),
            )
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let scalars = self
            .scalars
            .par_iter()
            .map(|x| {
                x.iter()
                    .map(|limb| witness.get_target(*limb))
                    .map(|x| F::as_canonical_u64(&x) as u32)
                    .collect::<Vec<_>>()
            })
            .map(BigUint::new)
            .collect::<Vec<_>>();
        assert_eq!(scalars.len(), 256);

        let points = self
            .points
            .par_iter()
            .map(|point| {
                let x = field_limbs_to_biguint(&witness.get_targets(&point.x));
                let y = field_limbs_to_biguint(&witness.get_targets(&point.y));
                AffinePoint::<Ed25519>::new(x, y)
            })
            .collect::<Vec<_>>();
        assert_eq!(points.len(), 256);

        let (tx, rx) = channel();
        for i in 0..256usize {
            let tx = tx.clone();
            let point = points[i].clone();
            let scalar = scalars[i].clone();
            rayon::spawn(move || {
                let res = point * scalar;
                tx.send((i, res)).unwrap();
            });
        }
        drop(tx);
        for (i, res) in rx.iter() {
            let res_limbs_x: [_; 16] = biguint_to_16_digits_field(&res.x, 16).try_into().unwrap();
            let res_limbs_y: [_; 16] = biguint_to_16_digits_field(&res.y, 16).try_into().unwrap();
            out_buffer.set_target_arr(&self.results[i].x, &res_limbs_x);
            out_buffer.set_target_arr(&self.results[i].y, &res_limbs_y);
        }
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        unimplemented!("SimpleScalarMulEd25519HintGenerator::serialize")
    }

    fn deserialize(
        _src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self> {
        unimplemented!("SimpleScalarMulEd25519HintGenerator::deserialize")
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
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

    #[test]
    fn test_scalar_generator() {
        type F = GoldilocksField;
        type E = GoldilocksCubicParameters;
        type C = PoseidonGoldilocksConfig;
        type S = EdDSAStark<F, E>;
        const D: usize = 2;

        let _ = env_logger::builder().is_test(true).try_init();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // Get virtual targets for scalars,  points
        let scalars_limbs = (0..256)
            .map(|_| {
                (0..8)
                    .map(|_| builder.add_virtual_target())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let points = (0..256)
            .map(|_| {
                let x = builder.add_virtual_target_arr();
                let y = builder.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        let expected_results = (0..256)
            .map(|_| {
                let x = builder.add_virtual_target_arr();
                let y = builder.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        // The scalar multiplications
        let results = builder.ed_scalar_mul_batch::<S, E, C>(&points, &scalars_limbs);

        // compare the results to the expected results
        for (result, expected) in results.iter().zip(expected_results.iter()) {
            for i in 0..16 {
                builder.connect(result.x[i], expected.x[i]);
                builder.connect(result.y[i], expected.y[i]);
            }
        }

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();

        let mut rng = thread_rng();
        let generator = Ed25519::generator();
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
            let scalar_limbs_iter = scalar.iter_u32_digits().map(F::from_canonical_u32);
            for (target, limb) in scalars_limbs[i].iter().zip(scalar_limbs_iter) {
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

    #[test]
    fn test_scalar_hint_generator() {
        type F = GoldilocksField;
        type C = PoseidonGoldilocksConfig;
        const D: usize = 2;

        let _ = env_logger::builder().is_test(true).try_init();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // Get virtual targets for scalars,  points
        let scalars_limbs = (0..256)
            .map(|_| {
                (0..8)
                    .map(|_| builder.add_virtual_target())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let points = (0..256)
            .map(|_| {
                let x = builder.add_virtual_target_arr();
                let y = builder.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        let expected_results = (0..256)
            .map(|_| {
                let x = builder.add_virtual_target_arr();
                let y = builder.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        // The scalar multiplications
        let results = builder.ed_scalar_mul_batch_hint(&points, &scalars_limbs);

        // compare the results to the expected results
        for (result, expected) in results.iter().zip(expected_results.iter()) {
            for i in 0..16 {
                builder.connect(result.x[i], expected.x[i]);
                builder.connect(result.y[i], expected.y[i]);
            }
        }

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();

        let mut rng = thread_rng();
        let generator = Ed25519::generator();
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
            let scalar_limbs_iter = scalar.iter_u32_digits().map(F::from_canonical_u32);
            for (target, limb) in scalars_limbs[i].iter().zip(scalar_limbs_iter) {
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
