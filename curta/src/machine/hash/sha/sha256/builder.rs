// use super::data::SHA256Data;
// use super::{INITIAL_HASH, ROUND_CONSTANTS};
// use crate::chip::arithmetic::expression::ArithmeticExpression;
// use crate::chip::memory::time::Time;
// use crate::chip::register::element::ElementRegister;
// use crate::chip::register::Register;
// use crate::chip::uint::operations::instruction::UintInstructions;
// use crate::chip::uint::register::U32Register;
// use crate::chip::uint::util::u32_to_le_field_bytes;
// use crate::chip::AirParameters;
// use crate::machine::builder::Builder;
// use crate::machine::bytes::builder::BytesBuilder;
// use crate::math::prelude::*;

// impl<L: AirParameters> BytesBuilder<L>
// where
//     L::Instruction: UintInstructions,
// {
//     pub fn sha_256_data(&mut self, num_rounds: usize) -> SHA256Data {
//         let num_round_element = self.constant(&L::Field::from_canonical_usize(num_rounds));

//         // let state = self.uninit_slice();
//         let initial_hash =
//             self.constant_array::<U32Register>(&INITIAL_HASH.map(u32_to_le_field_bytes));

//         let round_constant_values =
//             self.constant_array::<U32Register>(&ROUND_CONSTANTS.map(u32_to_le_field_bytes));

//         let round_constants = self.initialize_slice(
//             &round_constant_values,
//             &Time::zero(),
//             Some(num_round_element),
//         );

//         // let state = self.uninit_slice();

//         let process_id = self.alloc();

//         let index = self.alloc();
//         let is_preprocessing = self.alloc();
//         let w = self.uninit_slice();

//         let padded_messages = (0..num_rounds)
//             .map(|_| self.alloc_array_public::<U32Register>(16))
//             .collect::<Vec<_>>();
//         let dummy_entry = self.constant::<U32Register>(&[L::Field::ZERO; 4]);

//         let num_dummy_reads =
//             self.constant::<ElementRegister>(&L::Field::from_canonical_usize(64 - 16));

//         for (i, padded_message) in padded_messages.iter().enumerate() {
//             for (j, word) in padded_message.iter().enumerate().take(16) {
//                 self.store(&w.get(j), word, &Time::zero(), None);
//             }
//             self.store(
//                 &w.get(16),
//                 dummy_entry,
//                 &Time::zero(),
//                 Some(num_dummy_reads),
//             );
//         }

//         SHA256Data {
//             // state,
//             initial_hash,
//             round_constants,
//             w,
//             index,
//             is_preprocessing,
//             padded_messages,
//             process_id,
//         }
//     }

//     pub fn sha_256_preprocessing(&mut self, data: &SHA256Data) -> U32Register {
//         let w = &data.w;
//         let index = data.index;
//         let is_preprocessing = data.is_preprocessing;

//         // let index_shift = |j: usize| {
//         //     is_preprocessing.expr() * (index.expr() - L::Field::from_canonical_usize(j))
//         //         + is_preprocessing.not_expr() * L::Field::from_canonical_u8(16)
//         // };

//         // // Calculate the value:
//         // // s_0 = w_i_minus_15.rotate_right(7) ^ w_i_minus_15.rotate_right(18) ^ (w_i_minus_15 >> 3)
//         // let i_m_15 = self.expression(index_shift(15));
//         // let w_i_minus_15 = self.load(&w.get_at(i_m_15), &time);
//         // let w_i_minus_15_rotate_7 = self.rotate_right(w_i_minus_15, 7);
//         // let w_i_minus_15_rotate_18 = self.rotate_right(w_i_minus_15, 18);
//         // let w_i_minus_15_shr_3 = self.shr(w_i_minus_15, 3);

//         // let mut s_0 = self.xor(&w_i_minus_15_rotate_7, &w_i_minus_15_rotate_18);
//         // s_0 = self.xor(&s_0, &w_i_minus_15_shr_3);

//         // // Calculate the value:
//         // // s_1 = w_i_minus_2.rotate_right(17) ^ w_i_minus_2.rotate_right(19) ^ (w_i_minus_2 >> 10)
//         // let i_m_2 = self.expression(index_shift(2));
//         // let time_m_2_expr = self.expression(
//         //     is_preprocessing.expr() * (self.clk.expr() - L::Field::from_canonical_usize(2)),
//         // );
//         // let time_m_2 = Time::from_element(time_m_2_expr);
//         // let w_i_minus_2 = self.load(&w.get_at(i_m_2), &time_m_2);
//         // let w_i_minus_2_rotate_17 = self.rotate_right(w_i_minus_2, 17);
//         // let w_i_minus_2_rotate_19 = self.rotate_right(w_i_minus_2, 19);
//         // let w_i_minus_2_shr_10 = self.shr(w_i_minus_2, 10);

//         // let mut s_1 = self.xor(&w_i_minus_2_rotate_17, &w_i_minus_2_rotate_19);
//         // s_1 = self.xor(&s_1, &w_i_minus_2_shr_10);

//         // // Calculate the value:
//         // // w_i = w_i_minus_16 + s_0 + w_i_minus_7 + s_1
//         // let i_m_16 = self.expression(index_shift(16));
//         // let i_m_7 = self.expression(index_shift(7));
//         // let w_i_mimus_16 = self.load(&w.get_at(i_m_16), &time);
//         // let w_i_mimus_7 = self.load(&w.get_at(i_m_7), &time);
//         // let mut w_i_pre_process = self.add(w_i_mimus_16, s_0);
//         // w_i_pre_process = self.add(w_i_pre_process, w_i_mimus_7);
//         // w_i_pre_process = self.add(w_i_pre_process, s_1);

//         let i_idx = self.expression(
//             is_preprocessing.not_expr() * index.expr()
//                 + is_preprocessing.expr() * L::Field::from_canonical_u8(16),
//         );
//         let w_i_read = self.load(&w.get_at(i_idx), &Time::zero());

//         let w_i = w_i_read;

//         w_i
//     }
// }

// #[cfg(test)]
// mod tests {
//     use plonky2::field::goldilocks_field::GoldilocksField;
//     use plonky2::util::timing::TimingTree;
//     use serde::{Deserialize, Serialize};

//     use super::*;
//     use crate::chip::hash::sha::sha256::SHA256Gadget;
//     use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
//     use crate::chip::uint::operations::instruction::UintInstruction;
//     use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
//     use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};

//     #[derive(Clone, Debug, Serialize, Deserialize)]
//     pub struct SHAPreprocessingTest;

//     impl AirParameters for SHAPreprocessingTest {
//         type Field = GoldilocksField;
//         type CubicParams = GoldilocksCubicParameters;

//         type Instruction = UintInstruction;

//         const NUM_FREE_COLUMNS: usize = 186;
//         const EXTENDED_COLUMNS: usize = 120;
//     }

//     #[test]
//     fn test_sha256_preprocessing() {
//         type L = SHAPreprocessingTest;
//         type C = CurtaPoseidonGoldilocksConfig;
//         type Config = <C as CurtaConfig<2>>::GenericConfig;

//         let _ = env_logger::builder().is_test(true).try_init();
//         let mut timing = TimingTree::new("test_sha256_preprocessing", log::Level::Debug);

//         let mut builder = BytesBuilder::<L>::new();

//         let num_rounds = 1;

//         let data = builder.sha_256_data(num_rounds);

//         let w_i = builder.sha_256_preprocessing(&data);

//         let num_rows = 1 << (6 * num_rounds);
//         let stark = builder.build::<C, 2>(num_rows);

//         let writer = TraceWriter::new(&stark.air_data, num_rows);

//         let msg = b"abc";
//         // let expected_digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

//         let padded_msg_bytes = SHA256Gadget::pad(msg);
//         assert_eq!(padded_msg_bytes.len(), 64);
//         let padded_msg = padded_msg_bytes
//             .chunks_exact(4)
//             .map(|slice| u32::from_be_bytes(slice.try_into().unwrap()))
//             .map(u32_to_le_field_bytes::<GoldilocksField>)
//             .collect::<Vec<_>>();

//         assert_eq!(data.padded_messages.len(), 1);
//         writer.write_array(&data.padded_messages[0], &padded_msg, 0);

//         writer.write_global_instructions(&stark.air_data);

//         for i in 0..num_rows {
//             let k = i % 64;
//             writer.write(
//                 &data.process_id,
//                 &GoldilocksField::from_canonical_usize(i / 64),
//                 i,
//             );
//             writer.write(&data.index, &GoldilocksField::from_canonical_usize(k), i);
//             writer.write(
//                 &data.is_preprocessing,
//                 &GoldilocksField::from_canonical_u8((k >= 16) as u8),
//                 i,
//             );
//             writer.write_row_instructions(&stark.air_data, i);
//         }

//         let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
//         let proof = stark.prove(&trace, &public, &mut timing).unwrap();

//         stark.verify(proof.clone(), &public).unwrap();
//     }
// }
