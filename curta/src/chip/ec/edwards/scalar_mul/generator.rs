use core::fmt::Debug;
use std::sync::mpsc::channel;

use itertools::Itertools;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use serde::{Deserialize, Serialize};

use super::air::ScalarMulEd25519;
use super::gadget::EdScalarMulGadget;
use crate::chip::builder::AirTraceData;
use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField, Ed25519Parameters};
use crate::chip::ec::gadget::EllipticCurveWriter;
use crate::chip::ec::point::{AffinePoint, AffinePointRegister};
use crate::chip::ec::{EllipticCurve, EllipticCurveParameters};
use crate::chip::field::instruction::FpInstruction;
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::RegisterSerializable;
use crate::chip::trace::generator::ArithmeticGenerator;
use crate::chip::utils::{biguint_to_16_digits_field, biguint_to_bits_le, field_limbs_to_biguint};
use crate::chip::Chip;
use crate::math::extension::CubicParameters;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::gadget::StarkGadget;
use crate::plonky2::stark::proof::StarkProofTarget;
use crate::plonky2::stark::prover::StarkyProver;
use crate::plonky2::stark::verifier::set_stark_proof_target;
use crate::plonky2::stark::Starky;
use crate::utils::serde::{BufferRead, BufferWrite};

pub type EdDSAStark<F, E> = Starky<Chip<ScalarMulEd25519<F, E>>>;

const AFFINE_POINT_TARGET_NUM_LIMBS: usize = 16;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffinePointTarget {
    pub x: [Target; AFFINE_POINT_TARGET_NUM_LIMBS],
    pub y: [Target; AFFINE_POINT_TARGET_NUM_LIMBS],
}

pub trait ScalarMulEd25519Gadget<F: RichField + Extendable<D>, const D: usize> {
    fn ed_scalar_mul_batch<E: CubicParameters<F>, C: CurtaConfig<D, F = F, FE = F::Extension>>(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<Target>],
    ) -> Vec<AffinePointTarget>;

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
    fn ed_scalar_mul_batch<E: CubicParameters<F>, C: CurtaConfig<D, F = F, FE = F::Extension>>(
        &mut self,
        points: &[AffinePointTarget],
        scalars: &[Vec<Target>],
    ) -> Vec<AffinePointTarget> {
        // Input results
        let results = (0..256)
            .map(|_| {
                let x = self.add_virtual_target_arr();
                let y = self.add_virtual_target_arr();
                AffinePointTarget { x, y }
            })
            .collect::<Vec<_>>();

        let (
            air,
            trace_data,
            gadget,
            scalars_limbs_input,
            input_points,
            output_points,
            (set_last, set_bit),
        ) = ScalarMulEd25519::<F, E>::air();

        let mut public_input_target_option = vec![None as Option<Target>; 256 * (8 + 2 * 32)];
        for (scalar_register, scalar_target) in scalars_limbs_input.iter().zip_eq(scalars.iter()) {
            let (s_0, s_1) = scalar_register.register().get_range();
            let scalar_targets = scalar_target.iter().map(|x| Some(*x)).collect::<Vec<_>>();
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
                .collect::<Vec<_>>();
            public_input_target_option[p_x_0..p_y_1].copy_from_slice(&point_targets);
        }

        for (point_register, point_target) in output_points.iter().zip(results.iter()) {
            let (p_x_0, p_x_1) = point_register.x.register().get_range();
            let (p_y_0, p_y_1) = point_register.y.register().get_range();
            assert_eq!(p_x_1, p_y_0, "x and y registers must be consecutive");
            let point_targets = point_target
                .x
                .iter()
                .chain(point_target.y.iter())
                .map(|v| Some(*v))
                .collect::<Vec<_>>();
            public_input_target_option[p_x_0..p_y_1].copy_from_slice(&point_targets);
        }

        let public_input_target = public_input_target_option
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();

        let stark = Starky::new(air);
        let config = StarkyConfig::<C, D>::standard_fast_config(1 << 16);
        let proof_target = self.add_virtual_stark_proof(&stark, &config);

        self.verify_stark_proof(&config, &stark, &proof_target, &public_input_target);

        let generator = SimpleScalarMulEd25519Generator::<F, E, C, D> {
            gadget,
            points: points.to_vec(),
            scalars: scalars.to_vec(),
            results: results.clone(),
            set_last,
            set_bit,
            config,
            stark,
            trace_data,
            proof_target,
            scalars_limbs_input,
            input_points,
            output_points,
        };

        self.add_simple_generator(generator);

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SimpleScalarMulEd25519Generator<
    F: RichField + Extendable<D>,
    E: CubicParameters<F>,
    C: CurtaConfig<D, F = F>,
    const D: usize,
> {
    gadget: EdScalarMulGadget<F, Ed25519Parameters>,
    points: Vec<AffinePointTarget>,
    scalars: Vec<Vec<Target>>, // 32-byte limbs
    results: Vec<AffinePointTarget>,
    set_last: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
    set_bit: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
    config: StarkyConfig<C, D>,
    stark: Starky<Chip<ScalarMulEd25519<F, E>>>,
    trace_data: AirTraceData<ScalarMulEd25519<F, E>>,
    proof_target: StarkProofTarget<D>,
    scalars_limbs_input: Vec<ArrayRegister<ElementRegister>>,
    input_points: Vec<AffinePointRegister<Ed25519>>,
    output_points: Vec<AffinePointRegister<Ed25519>>,
}

impl<
        F: RichField + Extendable<D>,
        E: CubicParameters<F>,
        C: CurtaConfig<D, F = F>,
        const D: usize,
    > SimpleScalarMulEd25519Generator<F, E, C, D>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gadget: EdScalarMulGadget<F, Ed25519Parameters>,
        points: Vec<AffinePointTarget>,
        scalars: Vec<Vec<Target>>,
        results: Vec<AffinePointTarget>,
        set_last: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
        set_bit: AirInstruction<F, FpInstruction<Ed25519BaseField>>,
        config: StarkyConfig<C, D>,
        stark: Starky<Chip<ScalarMulEd25519<F, E>>>,
        trace_data: AirTraceData<ScalarMulEd25519<F, E>>,
        proof_target: StarkProofTarget<D>,
        scalars_limbs_input: Vec<ArrayRegister<ElementRegister>>,
        input_points: Vec<AffinePointRegister<Ed25519>>,
        output_points: Vec<AffinePointRegister<Ed25519>>,
    ) -> Self {
        Self {
            gadget,
            points,
            scalars,
            results,
            set_last,
            set_bit,
            config,
            stark,
            trace_data,
            proof_target,
            scalars_limbs_input,
            input_points,
            output_points,
        }
    }

    pub fn id() -> String {
        "SimpleScalarMulEd25519Generator".to_string()
    }
}

impl<
        F: RichField + Extendable<D>,
        E: CubicParameters<F>,
        C: CurtaConfig<D, F = F, FE = F::Extension>,
        const D: usize,
    > SimpleGenerator<F, D> for SimpleScalarMulEd25519Generator<F, E, C, D>
{
    fn id(&self) -> String {
        Self::id()
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
        // Generate the trace
        let trace_generator = ArithmeticGenerator::new(self.trace_data.clone(), 1 << 16);

        let writer = trace_generator.new_writer();

        let scalars = self
            .scalars
            .par_iter()
            .zip(self.scalars_limbs_input.par_iter())
            .map(|(x, register)| {
                let limbs = x
                    .iter()
                    .zip(register.iter())
                    .map(|(limb, reg)| (witness.get_target(*limb), reg))
                    .map(|(x, reg)| {
                        writer.write(&reg, &x, 0);
                        F::as_canonical_u64(&x) as u32
                    })
                    .collect::<Vec<_>>();

                limbs
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

        // Write the public inputs
        for (register, point) in self.input_points.iter().zip(points.iter()) {
            writer.write_ec_point(register, point, 0);
        }

        let (tx, rx) = channel();
        let res = self.gadget.double_and_add_gadget.result;
        let temp = self.gadget.double_and_add_gadget.temp;
        let scalar_bit = self.gadget.double_and_add_gadget.bit;
        let nb_bits = Ed25519::nb_scalar_bits();
        let tx_vec = (0..256).map(|_| tx.clone()).collect::<Vec<_>>();
        tx_vec.into_par_iter().enumerate().for_each(|(k, tx)| {
            let starting_row = 256 * k;
            writer.write_ec_point(&res, &Ed25519::neutral(), starting_row);
            writer.write_ec_point(&temp, &points[k], starting_row);
            let scalar_bits = biguint_to_bits_le(&scalars[k], nb_bits);
            for (i, bit) in scalar_bits.iter().enumerate() {
                let f_bit = F::from_canonical_u8(*bit as u8);
                writer.write(&scalar_bit, &f_bit, starting_row + i);
                writer.write_row_instructions(&trace_generator.air_data, starting_row + i);
            }
            tx.send((k, &points[k] * &scalars[k])).unwrap();
        });
        drop(tx);
        for (i, res) in rx.iter() {
            writer.write_ec_point(&self.output_points[i], &res, 0);
            let res_limbs_x: [_; 16] = biguint_to_16_digits_field(&res.x, 16).try_into().unwrap();
            let res_limbs_y: [_; 16] = biguint_to_16_digits_field(&res.y, 16).try_into().unwrap();
            out_buffer.set_target_arr(&self.results[i].x, &res_limbs_x);
            out_buffer.set_target_arr(&self.results[i].y, &res_limbs_y);
        }
        for j in (0..(1 << 16)).rev() {
            writer.write_instruction(&self.set_last, j);
            writer.write_instruction(&self.set_bit, j);
        }

        let public_inputs: Vec<_> = writer.public.read().unwrap().clone();

        let proof = StarkyProver::<F, C, D>::prove(
            &self.config,
            &self.stark,
            &trace_generator,
            &public_inputs,
        )
        .unwrap();

        set_stark_proof_target(out_buffer, &self.proof_target, &proof);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        let data = bincode::serialize(&self).unwrap();
        dst.write_bytes(&data)
    }

    fn deserialize(
        src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self> {
        let bytes = src.read_bytes()?;
        let data = bincode::deserialize(&bytes).unwrap();
        Ok(data)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn id() -> String {
        "SimpleScalarMulEd25519HintGenerator".to_string()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for SimpleScalarMulEd25519HintGenerator<F, D>
{
    fn id(&self) -> String {
        Self::id()
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
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        let data = bincode::serialize(&self).unwrap();
        dst.write_bytes(&data)
    }

    fn deserialize(
        src: &mut plonky2::util::serialization::Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self> {
        let bytes = src.read_bytes()?;
        let data = bincode::deserialize(&bytes).unwrap();
        Ok(data)
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
    use crate::chip::ec::EllipticCurve;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::plonky2::stark::config::CurtaPoseidonGoldilocksConfig;

    #[test]
    fn test_ed_generator_scalar_mul() {
        type F = GoldilocksField;
        type E = GoldilocksCubicParameters;
        type SC = CurtaPoseidonGoldilocksConfig;
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
        let results = builder.ed_scalar_mul_batch::<E, SC>(&points, &scalars_limbs);

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
        let generator = Ed25519::ec_generator();
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
        let generator = Ed25519::ec_generator();
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
