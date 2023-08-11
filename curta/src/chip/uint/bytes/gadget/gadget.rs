// use alloc::sync::Arc;
// use std::sync::Mutex;

// use plonky2::field::extension::Extendable;
// use plonky2::hash::hash_types::RichField;
// use plonky2::iop::target::Target;
// use plonky2::plonk::circuit_builder::CircuitBuilder;
// use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

// use super::air::{ByteGadgetParameters, NUM_BYTE_GADGET_COLUMNS};
// use super::generator::BytesLookupGenerator;
// use crate::chip::builder::AirBuilder;
// use crate::chip::trace::generator::ArithmeticGenerator;
// use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
// use crate::chip::uint::bytes::lookup_table::table::ByteLookupTable;
// use crate::chip::uint::bytes::operations::value::ByteOperation;
// use crate::chip::uint::bytes::register::ByteRegister;
// use crate::chip::AirParameters;

// #[derive(Debug, Clone, Copy)]
// pub struct ByteTarget(pub Target);

// pub trait CircuitBuilderBytes<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> {
//     fn set_byte_operation(
//         &mut self,
//         operation: ByteOperation<Target>,
//         gadget: &mut BytesGadget<F, E, D>,
//     );

//     fn add_virtual_byte_target_unsafe(&mut self, gadget: &mut BytesGadget<F, E, D>) -> ByteTarget;

//     fn add_virtual_byte_target(&mut self, gadget: &mut BytesGadget<F, E, D>) -> ByteTarget;

//     fn and_bytes(
//         &mut self,
//         a: ByteTarget,
//         b: ByteTarget,
//         gadget: &mut BytesGadget<F, E, D>,
//     ) -> ByteTarget;
//     fn xor_bytes(
//         &mut self,
//         a: ByteTarget,
//         b: ByteTarget,
//         gadget: &mut BytesGadget<F, E, D>,
//     ) -> ByteTarget;
//     fn not_bytes(&mut self, a: ByteTarget, gadget: &mut BytesGadget<F, E, D>) -> ByteTarget;
//     fn shr_bytes(
//         &mut self,
//         a: ByteTarget,
//         shift: u8,
//         gadget: &mut BytesGadget<F, E, D>,
//     ) -> ByteTarget;

//     fn register_byte_operations<C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone>(
//         &mut self,
//         gadget: BytesGadget<F, E, D>,
//     ) where
//         C::Hasher: AlgebraicHasher<F>;
// }

// use crate::math::prelude::*;
// use crate::plonky2::stark::config::StarkyConfig;
// use crate::plonky2::stark::gadget::StarkGadget;
// use crate::plonky2::stark::generator::simple::SimpleStarkWitnessGenerator;
// use crate::plonky2::stark::Starky;
// #[derive(Debug, Clone)]
// pub struct BytesGadget<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> {
//     operations: Vec<ByteOperation<Target>>,
//     air_operations: Vec<ByteOperation<ByteRegister>>,
//     lookup_operations: ByteLookupOperations,
//     table: Arc<Mutex<ByteLookupTable<F>>>,
//     air_builder: AirBuilder<ByteGadgetParameters<F, E, D>>,
// }

// impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> BytesGadget<F, E, D> {
//     pub fn new() -> Self {
//         let mut builder = AirBuilder::new();
//         let (operations, table) = builder.byte_operations();

//         Self {
//             operations: Vec::new(),
//             air_operations: Vec::new(),
//             air_builder: builder,
//             lookup_operations: operations,
//             table: Arc::new(Mutex::new(table)),
//         }
//     }
// }

// impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize>
//     CircuitBuilderBytes<F, E, D> for CircuitBuilder<F, D>
// {
//     fn set_byte_operation(
//         &mut self,
//         operation: ByteOperation<Target>,
//         gadget: &mut BytesGadget<F, E, D>,
//     ) {
//         let air_operation = gadget
//             .air_builder
//             .alloc_public_byte_operation_from_template(&operation);
//         gadget
//             .air_builder
//             .set_public_inputs_byte_operation(&air_operation, &mut gadget.lookup_operations);
//         gadget.air_operations.push(air_operation);
//         gadget.operations.push(operation);
//     }

//     fn add_virtual_byte_target_unsafe(&mut self, _gadget: &mut BytesGadget<F, E, D>) -> ByteTarget {
//         ByteTarget(self.add_virtual_target())
//     }

//     fn add_virtual_byte_target(&mut self, gadget: &mut BytesGadget<F, E, D>) -> ByteTarget {
//         let target = self.add_virtual_target();
//         let range_op = ByteOperation::Range(target);
//         self.set_byte_operation(range_op, gadget);
//         ByteTarget(target)
//     }

//     fn and_bytes(
//         &mut self,
//         a: ByteTarget,
//         b: ByteTarget,
//         gadget: &mut BytesGadget<F, E, D>,
//     ) -> ByteTarget {
//         let result = self.add_virtual_target();
//         let and_op = ByteOperation::And(a.0, b.0, result);
//         self.set_byte_operation(and_op, gadget);
//         ByteTarget(result)
//     }

//     fn xor_bytes(
//         &mut self,
//         a: ByteTarget,
//         b: ByteTarget,
//         gadget: &mut BytesGadget<F, E, D>,
//     ) -> ByteTarget {
//         let result = self.add_virtual_target();
//         let xor_op = ByteOperation::Xor(a.0, b.0, result);
//         self.set_byte_operation(xor_op, gadget);
//         ByteTarget(result)
//     }

//     fn not_bytes(&mut self, a: ByteTarget, gadget: &mut BytesGadget<F, E, D>) -> ByteTarget {
//         let result = self.add_virtual_target();
//         let not_op = ByteOperation::Not(a.0, result);
//         self.set_byte_operation(not_op, gadget);
//         ByteTarget(result)
//     }

//     fn shr_bytes(
//         &mut self,
//         a: ByteTarget,
//         shift: u8,
//         gadget: &mut BytesGadget<F, E, D>,
//     ) -> ByteTarget {
//         let result = self.add_virtual_target();
//         let shr_op = ByteOperation::ShrConst(a.0, shift, result);
//         self.set_byte_operation(shr_op, gadget);
//         ByteTarget(result)
//     }

//     fn register_byte_operations<C>(&mut self, gadget: BytesGadget<F, E, D>)
//     where
//         C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone,
//         C::Hasher: AlgebraicHasher<F>,
//     {
//         // Register the operations into the table
//         let BytesGadget {
//             operations,
//             air_operations,
//             mut lookup_operations,
//             table,
//             mut air_builder,
//         } = gadget;

//         let a = air_builder.alloc::<ByteRegister>();
//         let b = air_builder.alloc::<ByteRegister>();
//         let a_range = ByteOperation::Range(a);
//         let b_range = ByteOperation::Range(b);
//         air_builder.set_byte_operation(&a_range, &mut lookup_operations);
//         air_builder.set_byte_operation(&b_range, &mut lookup_operations);

//         let mut table_write = table.lock().unwrap();
//         air_builder.register_byte_lookup(lookup_operations, &mut table_write);
//         drop(table_write);

//         // Build the air
//         let air = air_builder.build();

//         let trace_generator = ArithmeticGenerator::<ByteGadgetParameters<F, E, D>>::new(&air);

//         let public_input_target = operations
//             .iter()
//             .flat_map(|op| op.all_targets())
//             .collect::<Vec<_>>();

//         // Initialize the byte operation generator
//         let byte_generator = BytesLookupGenerator::new(
//             operations,
//             air_operations,
//             trace_generator.clone(),
//             table,
//             air.clone(),
//         );
//         self.add_simple_generator(byte_generator);

//         let stark = Starky::<_, NUM_BYTE_GADGET_COLUMNS>::new(air);
//         let config = StarkyConfig::<F, C, D>::standard_fast_config(
//             ByteGadgetParameters::<F, E, D>::num_rows(),
//         );
//         let virtual_proof = self.add_virtual_stark_proof(&stark, &config);
//         // self.verify_stark_proof(&config, &stark, virtual_proof.clone(), &public_input_target);

//         let stark_generator = SimpleStarkWitnessGenerator::new(
//             config,
//             stark,
//             virtual_proof,
//             public_input_target,
//             trace_generator,
//         );

//         // self.add_simple_generator(stark_generator);
//     }
// }

// #[cfg(test)]
// mod tests {
//     use plonky2::field::goldilocks_field::GoldilocksField;
//     use plonky2::iop::witness::{PartialWitness, WitnessWrite};
//     use plonky2::plonk::circuit_builder::CircuitBuilder;
//     use plonky2::plonk::circuit_data::CircuitConfig;
//     use plonky2::plonk::config::PoseidonGoldilocksConfig;
//     use plonky2::timed;
//     use plonky2::util::timing::TimingTree;
//     use rand::{thread_rng, Rng};

//     use super::*;
//     use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

//     #[test]
//     fn test_byte_generator_gadget() {
//         type F = GoldilocksField;
//         type E = GoldilocksCubicParameters;
//         type C = PoseidonGoldilocksConfig;
//         const D: usize = 2;

//         let num_ops = 10000;

//         let _ = env_logger::builder().is_test(true).try_init();

//         let config = CircuitConfig::standard_recursion_config();
//         let mut builder = CircuitBuilder::<F, D>::new(config);
//         let mut gadget = BytesGadget::<F, E, D>::new();

//         let mut a_targets = Vec::new();
//         let mut b_targets = Vec::new();

//         let mut and_expected_targets = Vec::new();
//         let mut xor_expected_targets = Vec::new();
//         let mut shr_expected_targets = Vec::new();
//         let mut a_not_expected_targets = Vec::new();
//         for _ in 0..num_ops {
//             let a = builder.add_virtual_byte_target(&mut gadget);
//             let b = builder.add_virtual_byte_target(&mut gadget);
//             a_targets.push(a);
//             b_targets.push(b);

//             let a_and_b = builder.and_bytes(a, b, &mut gadget);
//             let and_expected = builder.add_virtual_byte_target_unsafe(&mut gadget);
//             builder.connect(a_and_b.0, and_expected.0);
//             and_expected_targets.push(and_expected);
//             let a_xor_b = builder.xor_bytes(a, b, &mut gadget);
//             let xor_expected = builder.add_virtual_byte_target_unsafe(&mut gadget);
//             builder.connect(a_xor_b.0, xor_expected.0);
//             xor_expected_targets.push(xor_expected);
//             let a_shr = builder.shr_bytes(a, 3, &mut gadget);
//             let shr_expected = builder.add_virtual_byte_target_unsafe(&mut gadget);
//             builder.connect(a_shr.0, shr_expected.0);
//             shr_expected_targets.push(shr_expected);
//             let a_not = builder.not_bytes(a, &mut gadget);
//             let a_not_expected = builder.add_virtual_byte_target_unsafe(&mut gadget);
//             builder.connect(a_not.0, a_not_expected.0);
//             a_not_expected_targets.push(a_not_expected);
//         }

//         builder.register_byte_operations::<C>(gadget);

//         let data = builder.build::<C>();
//         let mut pw = PartialWitness::new();

//         let mut rng = thread_rng();
//         let to_field = |x| F::from_canonical_u8(x);
//         for (k, (a, b)) in a_targets.into_iter().zip(b_targets).enumerate() {
//             let a_val = rng.gen::<u8>();
//             let b_val = rng.gen::<u8>();
//             pw.set_target(a.0, to_field(a_val));
//             pw.set_target(b.0, to_field(b_val));
//             pw.set_target(and_expected_targets[k].0, to_field(a_val & b_val));
//             pw.set_target(xor_expected_targets[k].0, to_field(a_val ^ b_val));
//             pw.set_target(shr_expected_targets[k].0, to_field(a_val >> 3));
//             pw.set_target(a_not_expected_targets[k].0, to_field(!a_val));
//         }

//         let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
//         let recursive_proof = timed!(
//             timing,
//             "Generate proof",
//             plonky2::plonk::prover::prove(&data.prover_only, &data.common, pw, &mut timing)
//         )
//         .unwrap();
//         timing.print();
//         data.verify(recursive_proof).unwrap();
//     }

//     #[test]
//     fn test_bit_equivalent() {
//         type F = GoldilocksField;
//         type C = PoseidonGoldilocksConfig;
//         const D: usize = 2;

//         let num_ops = 1000;

//         let _ = env_logger::builder().is_test(true).try_init();

//         let config = CircuitConfig::standard_recursion_config();
//         let mut builder = CircuitBuilder::<F, D>::new(config);

//         let mut a_targets = Vec::new();
//         let mut b_targets = Vec::new();

//         let mut and_expected_targets = Vec::new();
//         let mut xor_expected_targets = Vec::new();
//         let mut shr_expected_targets = Vec::new();
//         let mut a_not_expected_targets = Vec::new();

//         for _ in 0..num_ops {
//             let a: [_; 8] = core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             let b: [_; 8] = core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             a_targets.push(a);
//             b_targets.push(b);

//             let a_and_b: [_; 8] = core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             let and_expected: [_; 8] =
//                 core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             for ((a_bit, b_bit), a_and_b_bit) in a.iter().zip(b.iter()).zip(a_and_b.iter()) {
//                 let a_bit = *a_bit;
//                 let b_bit = *b_bit;
//                 let a_and_b = builder.and(a_bit, b_bit);
//                 builder.connect(a_and_b.target, a_and_b_bit.target);
//             }
//             for (res, exp) in a_and_b.iter().zip(and_expected.iter()) {
//                 builder.connect(res.target, exp.target);
//             }
//             and_expected_targets.push(and_expected);

//             let a_xor_b: [_; 8] = core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             let xor_expected: [_; 8] =
//                 core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             for (res, exp) in a_xor_b.iter().zip(xor_expected.iter()) {
//                 builder.connect(res.target, exp.target);
//             }
//             xor_expected_targets.push(xor_expected);
//             let a_shr: [_; 8] = core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             let shr_expected: [_; 8] =
//                 core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             for (res, exp) in a_shr.iter().zip(shr_expected.iter()) {
//                 builder.connect(res.target, exp.target);
//             }
//             shr_expected_targets.push(shr_expected);
//             let a_not: [_; 8] = core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             let a_not_expected: [_; 8] =
//                 core::array::from_fn(|_| builder.add_virtual_bool_target_safe());
//             for (res, exp) in a_not.iter().zip(a_not_expected.iter()) {
//                 builder.connect(res.target, exp.target);
//             }
//             a_not_expected_targets.push(a_not_expected);
//         }

//         let data = builder.build::<C>();
//         let mut pw = PartialWitness::new();

//         let mut rng = thread_rng();
//         for (k, (a, b)) in a_targets.into_iter().zip(b_targets).enumerate() {
//             let a_val = rng.gen::<u8>();
//             let b_val = rng.gen::<u8>();
//             for j in 0..8 {
//                 pw.set_bool_target(a[j], (a_val >> j) & 1 == 1);
//                 pw.set_bool_target(b[j], (b_val >> j) & 1 == 1);
//                 pw.set_bool_target(and_expected_targets[k][j], ((a_val & b_val) >> j) & 1 == 1);
//                 pw.set_bool_target(xor_expected_targets[k][j], ((a_val ^ b_val) >> j) & 1 == 1);
//                 pw.set_bool_target(shr_expected_targets[k][j], ((a_val >> 3) >> j) & 1 == 1);
//                 pw.set_bool_target(a_not_expected_targets[k][j], ((!a_val) >> j) & 1 == 1);
//             }
//         }

//         let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
//         let recursive_proof = timed!(
//             timing,
//             "Generate proof",
//             plonky2::plonk::prover::prove(&data.prover_only, &data.common, pw, &mut timing)
//         )
//         .unwrap();
//         timing.print();
//         data.verify(recursive_proof).unwrap();
//     }
// }
