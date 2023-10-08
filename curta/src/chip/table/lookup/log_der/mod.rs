//! This module implements a lookup argument based on the logarithmic derivative as in
//! https://eprint.iacr.org/2022/1530.pdf
//!

use self::table::LogLookupTable;
use self::values::LogLookupValues;

pub mod constraint;
pub mod table;
pub mod trace;
pub mod values;

// #[cfg(test)]
// mod tests {
//     use plonky2::field::types::Sample;
//     use rand::{thread_rng, Rng};

//     use super::*;
//     use crate::chip::builder::tests::*;
//     use crate::chip::register::Register;
//     use crate::chip::AirParameters;
//     use crate::math::extension::cubic::element::CubicElement;

//     #[derive(Debug, Clone, Serialize, Deserialize)]
//     struct LookupTest<const N: usize, const M: usize>;

//     impl<const N: usize, const M: usize> AirParameters for LookupTest<N, M> {
//         type Field = GoldilocksField;
//         type CubicParams = GoldilocksCubicParameters;

//         const NUM_FREE_COLUMNS: usize = M + 2 * N;
//         const EXTENDED_COLUMNS: usize = (3 * (M / 2)) + 3 * N + 2 * 3;

//         type Instruction = EmptyInstruction<GoldilocksField>;
//     }

//     #[test]
//     fn test_lookup() {
//         type L = LookupTest<N, M>;
//         type F = GoldilocksField;
//         type SC = PoseidonGoldilocksStarkConfig;
//         const N: usize = 29;
//         const M: usize = 10;

//         let mut builder = AirBuilder::<L>::new();

//         let table_values = builder
//             .alloc_array::<ElementRegister>(N)
//             .into_iter()
//             .collect::<Vec<_>>();
//         let values = builder
//             .alloc_array::<ElementRegister>(M)
//             .into_iter()
//             .collect::<Vec<_>>();

//         let multiplicities = builder.element_lookup(&table_values, &values);

//         let (air, trace_data) = builder.build();

//         let num_rows = 1 << 16;
//         let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
//         let writer = generator.new_writer();

//         // Set the table vals
//         for i in 0..num_rows {
//             let table_vals = [GoldilocksField::rand(); N];
//             for (reg, val) in table_values.iter().zip(table_vals) {
//                 writer.write(reg, &val, i);
//             }
//         }

//         let mut rng = thread_rng();
//         // Se the lookup vals
//         for i in 0..num_rows {
//             let j_vals = [rng.gen_range(0..num_rows); M];
//             let k_vals = [rng.gen_range(0..N); M];
//             for (value, (&j, &k)) in values.iter().zip(j_vals.iter().zip(k_vals.iter())) {
//                 let val = writer.read(&table_values[k], j);
//                 let mult_value = writer.read(&multiplicities.get(k), j);
//                 writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
//                 writer.write(value, &val, i);
//             }
//         }

//         let stark = Starky::from_chip(air);

//         let config = SC::standard_fast_config(num_rows);

//         // Generate proof and verify as a stark
//         test_starky(&stark, &config, &generator, &[]);

//         // Test the recursive proof.
//         test_recursive_starky(stark, config, generator, &[]);
//     }

//     #[derive(Debug, Clone, Serialize, Deserialize)]
//     struct CubicLookupTest<const N: usize, const M: usize>;

//     impl<const N: usize, const M: usize> AirParameters for CubicLookupTest<N, M> {
//         type Field = GoldilocksField;
//         type CubicParams = GoldilocksCubicParameters;

//         const NUM_FREE_COLUMNS: usize = 3 * M + 2 * 3 * N;
//         const EXTENDED_COLUMNS: usize = (3 * (3 * M / 2)) + 3 * 3 * N + 2 * 3;

//         type Instruction = EmptyInstruction<GoldilocksField>;
//     }

//     #[test]
//     fn test_cubic_lookup() {
//         type L = CubicLookupTest<N, M>;
//         type F = GoldilocksField;
//         type SC = PoseidonGoldilocksStarkConfig;
//         const N: usize = 4;
//         const M: usize = 102;

//         let mut builder = AirBuilder::<L>::new();

//         let table_values = builder
//             .alloc_array::<CubicRegister>(N)
//             .into_iter()
//             .collect::<Vec<_>>();
//         let values = builder
//             .alloc_array::<CubicRegister>(M)
//             .into_iter()
//             .collect::<Vec<_>>();

//         let multiplicities = builder.cubic_lookup(&table_values, &values);

//         let (air, trace_data) = builder.build();

//         let num_rows = 1 << 16;
//         let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
//         let writer = generator.new_writer();

//         // Set the table vals
//         for i in 0..num_rows {
//             let table_vals = [CubicElement::from_slice(&[GoldilocksField::rand(); 3]); N];
//             for (reg, val) in table_values.iter().zip(table_vals) {
//                 writer.write(reg, &val, i);
//             }
//         }

//         let mut rng = thread_rng();
//         // Se the lookup vals
//         for i in 0..num_rows {
//             let j_vals = [rng.gen_range(0..num_rows); M];
//             let k_vals = [rng.gen_range(0..N); M];
//             for (value, (&j, &k)) in values.iter().zip(j_vals.iter().zip(k_vals.iter())) {
//                 let val = writer.read(&table_values[k], j);
//                 let mult_value = writer.read(&multiplicities.get(k), j);
//                 writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
//                 writer.write(value, &val, i);
//             }
//         }

//         let stark = Starky::from_chip(air);

//         let config = SC::standard_fast_config(num_rows);

//         // Generate proof and verify as a stark
//         test_starky(&stark, &config, &generator, &[]);

//         // Test the recursive proof.
//         test_recursive_starky(stark, config, generator, &[]);
//     }

//     #[test]
//     fn test_cubic_lookup_with_public_values() {
//         type L = CubicLookupTest<N, M>;
//         type F = GoldilocksField;
//         type SC = PoseidonGoldilocksStarkConfig;
//         const N: usize = 4;
//         const M: usize = 102;
//         const PUB: usize = 10;
//         const GLOB: usize = 2;

//         let mut builder = AirBuilder::<L>::new();

//         let table_values = builder
//             .alloc_array::<CubicRegister>(N)
//             .into_iter()
//             .collect::<Vec<_>>();
//         let trace_values = builder
//             .alloc_array::<CubicRegister>(M)
//             .into_iter()
//             .collect::<Vec<_>>();

//         let public_values = builder
//             .alloc_array_public::<CubicRegister>(PUB)
//             .into_iter()
//             .collect::<Vec<_>>();

//         let global_values = builder
//             .alloc_array_global::<CubicRegister>(GLOB)
//             .into_iter()
//             .collect::<Vec<_>>();

//         let values = trace_values
//             .iter()
//             .copied()
//             .chain(public_values.iter().copied())
//             .chain(global_values.iter().copied())
//             .collect::<Vec<_>>();

//         // let multiplicities = builder.lookup_cubic_log_derivative(&table_values, &values);
//         let challenge = builder.alloc_challenge::<CubicRegister>();
//         let lookup_table = builder.lookup_table(&challenge, &table_values);
//         let lookup_values = builder.lookup_values(&challenge, &values);
//         assert_eq!(lookup_values.trace_values.len(), M);
//         assert_eq!(lookup_values.public_values.len(), PUB + GLOB);
//         let multiplicities = lookup_table.multiplicities;
//         builder.cubic_lookup_from_table_and_values(lookup_table, lookup_values);

//         let (air, trace_data) = builder.build();

//         let num_rows = 1 << 16;
//         let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
//         let writer = generator.new_writer();

//         // Set the table vals
//         for i in 0..num_rows {
//             let table_vals = [CubicElement::from_slice(&[GoldilocksField::rand(); 3]); N];
//             for (reg, val) in table_values.iter().zip(table_vals) {
//                 writer.write(reg, &val, i);
//             }
//         }

//         let mut rng = thread_rng();
//         // Set the lookup vals
//         for i in 0..num_rows {
//             let j_vals = [rng.gen_range(0..num_rows); M];
//             let k_vals = [rng.gen_range(0..N); M];
//             for (value, (&j, &k)) in values.iter().zip(j_vals.iter().zip(k_vals.iter())) {
//                 let val = writer.read(&table_values[k], j);
//                 let mult_value = writer.read(&multiplicities.get(k), j);
//                 writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
//                 writer.write(value, &val, i);
//             }
//         }

//         // Set the public values
//         let mut public_inputs: Vec<F> = vec![F::ZERO; air.num_public_values];
//         for public_value in public_values.iter() {
//             let j = rng.gen_range(0..num_rows);
//             let k = rng.gen_range(0..N);
//             let val = writer.read(&table_values[k], j);
//             public_value.assign_to_raw_slice(&mut public_inputs, &val);
//             let mult_value = writer.read(&multiplicities.get(k), j);
//             writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
//         }

//         for global_value in global_values.iter() {
//             let j = rng.gen_range(0..num_rows);
//             let k = rng.gen_range(0..N);
//             let val = writer.read(&table_values[k], j);
//             writer.write(global_value, &val, 0);
//             let mult_value = writer.read(&multiplicities.get(k), j);
//             writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
//         }

//         let stark = Starky::from_chip(air);

//         let config = SC::standard_fast_config(num_rows);

//         // Generate proof and verify as a stark
//         test_starky(&stark, &config, &generator, &public_inputs);

//         // Test the recursive proof.
//         test_recursive_starky(stark, config, generator, &public_inputs);
//     }
// }
