pub mod builder_gadget;
pub mod generator;

use core::slice::SlicePattern;

use itertools::Itertools;

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::chip::table::bus::global::Bus;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::{ByteArrayRegister, U64Register, U8Register};
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub type U64Value<T> = <U64Register as Register>::Value<T>;

#[derive(Debug, Clone)]
pub struct BLAKE2BGadget {
    pub t: U64Register,
    pub m: ArrayRegister<U64Register>,
    pub h: ArrayRegister<U64Register>,
    pub end_bit: BitRegister,

    // Public values
    pub msg_chunks: ArrayRegister<U64Register>,
    pub initial_hash: ArrayRegister<U64Register>,
    pub initial_hash_compress: ArrayRegister<U64Register>,
    pub hash_state: ArrayRegister<U64Register>,
    pub inversion_const: U64Register,
}

#[derive(Debug, Clone)]
pub struct BLAKE2BPublicData<T> {
    pub hash_state: Vec<U64Value<T>>,
    pub end_bits: Vec<T>,
}

const INITIAL_HASH: [u64; 8] = [
    0x6a09e667f2bdc928,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

// Note that for this blake2b implementation, we don't support a key input and
// we assume that the output is 32 bytes
// So that means the initial hash entry to be
// 0x6a09e667f3bcc908 xor 0x01010020
const INITIAL_HASH_COMPRESS: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const INVERSION_CONST: u64 = 0xFFFFFFFFFFFFFFFF;

const SIGMA_LEN: usize = 10;
const SIGMA: [[usize; 16]; SIGMA_LEN] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

impl<L: AirParameters> AirBuilder<L> {
    pub fn process_blake2b(
        &mut self,
        clk: &ElementRegister,
        bus: &mut Bus<L::CubicParams>,
        bus_channel_idx: usize,
        operations: &mut ByteLookupOperations,
    ) -> BLAKE2BGadget
    where
        L::Instruction: U32Instructions,
    {
        // Registers to be written to
        let t = self.alloc::<U64Register>();
        let m = self.alloc_array::<U64Register>(16);
        let h = self.alloc_array::<U64Register>(8);
        let end_bit = self.alloc::<BitRegister>();
        let cycle_12_clk = self.alloc::<U8Register>();
        let cycle_12_end_bit = self.alloc::<BitRegister>();

        // Public values
        let msg_chunks = self.alloc_array_public::<U64Register>(16 * 512);
        let initial_hash = self.alloc_array_public::<U64Register>(8);
        let initial_hash_compress = self.alloc_array_public::<U64Register>(8);
        let hash_state = self.alloc_array_public::<U64Register>(8 * 512);
        let inversion_const = self.alloc_public::<U64Register>();
        let end_bits_public = self.alloc_array_public::<BitRegister>(512);

        // Get message chunk challenges
        let message_chunk_challenges =
            self.alloc_challenge_array::<CubicRegister>(U64Register::size_of() * 16 + 1);

        // Get hash state challenges
        let state_challenges =
            self.alloc_challenge_array::<CubicRegister>(U64Register::size_of() * 8 + 1);

        // Get a challenge for the end bit
        let end_bit_challenge = self.alloc_challenge_array::<CubicRegister>(2);

        // Put end bit values into the bus
        let clk_end_bit =
            self.accumulate_expressions(&end_bit_challenge, &[clk.expr(), end_bit.expr()]);

        // Put the end_bit in the bus at the end of each round
        self.input_to_bus_filtered(bus_channel_idx, clk_end_bit, cycle_12_end_bit.expr());

        // Constrain all other values of end_bit to zero
        self.assert_expression_zero(end_bit.expr() * cycle_12_end_bit.not_expr());

        // Put public hash state, end_bits, and all the msg chunk permutations into the bus
        for i in 0..512 {
            let state_digest = self.accumulate_public_expressions(
                &state_challenges,
                &[
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(
                        i * 12 + 11,
                    )),
                    hash_state.get_subarray(i * 8..i * 8 + 8).expr(),
                ],
            );

            bus.output_global_value(&state_digest);

            let bit_digest = self.accumulate_public_expressions(
                &end_bit_challenge,
                &[
                    ArithmeticExpression::from_constant(L::Field::from_canonical_usize(
                        i * 12 + 11,
                    )),
                    end_bits_public.get(i).expr(),
                ],
            );
            bus.output_global_value(&bit_digest);

            for k in 0..12 {
                let row = i * 12 + k;
                let sigma = SIGMA[k % SIGMA_LEN];
                let mut values = Vec::new();

                values.push(ArithmeticExpression::from_constant(
                    L::Field::from_canonical_usize(row),
                ));

                for index in sigma.iter() {
                    values.push(msg_chunks.get(i * 16 + index).expr());
                }

                let msg_digest = self
                    .accumulate_public_expressions(&message_chunk_challenges, values.as_slice());

                bus.output_global_value(&msg_digest);
            }
        }

        let clk_msg_digest = self.accumulate_public_expressions(&message_chunk_challenges, &[
            clk,
            m.get_subarray(0..16).expr(),
        ]);
        bus.input_value(&clk_msg_digest);

        for (h, init) in h.iter().zip(initial_hash.iter()) {
            self.set_to_expression_first_row(&h, init.expr());
        }

        let next_h = self.blake2b_compress(
            clk,
            &m,
            &h,
            &initial_hash_compress,
            &inversion_const,
            &t,
            operations,
            &end_bit,
        );

        let clk_hash_next =
            self.accumulate_expressions(&state_challenges, &[clk.expr(), next_h.expr()]);
        self.input_to_bus_filtered(bus_channel_idx, clk_hash_next, cycle_12_end_bit.expr());

        BLAKE2BGadget {
            t,
            m,
            h,
            end_bit,
            msg_chunks,
            initial_hash,
            initial_hash_compress,
            hash_state,
            inversion_const,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn blake2b_compress(
        &mut self,
        clk: &ElementRegister,
        m: &ArrayRegister<U64Register>,
        h: &ArrayRegister<U64Register>,
        iv_pub: &ArrayRegister<U64Register>,
        inversion_const_pub: &U64Register,
        t: &U64Register, // assumes t is not more than u64
        operations: &mut ByteLookupOperations,
        end_bit: &BitRegister,
    ) -> ArrayRegister<U64Register>
    where
        L::Instruction: U32Instructions,
    {
        let iv = self.alloc_array::<U64Register>(8);
        for i in 0..8 {
            self.set_to_expression_first_row(&iv.get(i), iv_pub.get(i).expr());
        }

        let inversion_const = self.alloc::<U64Register>();
        self.set_to_expression_first_row(&inversion_const, inversion_const_pub.expr());

        let mut V0 = h.get(0);
        let mut V1 = h.get(1);
        let mut V2 = h.get(2);
        let mut V3 = h.get(3);
        let mut V4 = h.get(4);
        let mut V5 = h.get(5);
        let mut V6 = h.get(6);
        let mut V7 = h.get(7);

        let mut V8 = iv.get(0);
        let mut V9 = iv.get(1);
        let mut V10 = iv.get(2);
        let mut V11 = iv.get(3);
        let mut V12 = iv.get(4);
        let mut V13 = iv.get(5);
        let mut V14 = iv.get(6);
        let mut V15 = iv.get(7);

        V12 = self.bitwise_xor(&V12, &t, operations);
        // We assume that t is no more than u64, so we don't modify V13

        let tmp = self.bitwise_xor(&V14, &inversion_const, operations);
        let mut V14New = self.alloc::<ByteArrayRegister<8>>();
        self.set_to_expression(
            &V14New,
            tmp.expr() * last_block_bit.expr() + V14.expr() * last_block_bit.not_expr(),
        );

        self.blake2b_mix(
            &mut V0,
            &mut V4,
            &mut V8,
            &mut V12,
            &m.get(0),
            &m.get(1),
            operations,
        );

        self.blake2b_mix(
            &mut V1,
            &mut V5,
            &mut V9,
            &mut V13,
            &m.get(2),
            &m.get(3),
            operations,
        );

        self.blake2b_mix(
            &mut V2,
            &mut V6,
            &mut V10,
            &mut V14New,
            &m.get(4),
            &m.get(5),
            operations,
        );

        self.blake2b_mix(
            &mut V3,
            &mut V7,
            &mut V11,
            &mut V15,
            &m.get(6),
            &m.get(7),
            operations,
        );

        self.blake2b_mix(
            &mut V0,
            &mut V5,
            &mut V10,
            &mut V15,
            &m.get(8),
            &m.get(9),
            operations,
        );

        self.blake2b_mix(
            &mut V1,
            &mut V6,
            &mut V11,
            &mut V12,
            &m.get(10),
            &m.get(11),
            operations,
        );

        self.blake2b_mix(
            &mut V2,
            &mut V7,
            &mut V8,
            &mut V13,
            &m.get(12),
            &m.get(13),
            operations,
        );

        self.blake2b_mix(
            &mut V3,
            &mut V4,
            &mut V9,
            &mut V14,
            &m.get(14),
            &m.get(15),
            operations,
        );

        let next_h = self.alloc_array::<U64Register>(8);

        let next_h_0_tmp = self.bitwise_xor(&h.get(0), &V0, operations);
        let next_h_1_tmp = self.bitwise_xor(&h.get(1), &V1, operations);
        let next_h_2_tmp = self.bitwise_xor(&h.get(2), &V2, operations);
        let next_h_3_tmp = self.bitwise_xor(&h.get(3), &V3, operations);
        let next_h_4_tmp = self.bitwise_xor(&h.get(4), &V4, operations);
        let next_h_5_tmp = self.bitwise_xor(&h.get(5), &V5, operations);
        let next_h_6_tmp = self.bitwise_xor(&h.get(6), &V6, operations);
        let next_h_7_tmp = self.bitwise_xor(&h.get(7), &V7, operations);

        self.set_bitwise_xor(&next_h_0_tmp, &V8, &next_h.get(0), operations);
        self.set_bitwise_xor(&next_h_1_tmp, &V9, &next_h.get(1), operations);
        self.set_bitwise_xor(&next_h_2_tmp, &V10, &next_h.get(2), operations);
        self.set_bitwise_xor(&next_h_3_tmp, &V11, &next_h.get(3), operations);
        self.set_bitwise_xor(&next_h_4_tmp, &V12, &next_h.get(4), operations);
        self.set_bitwise_xor(&next_h_5_tmp, &V13, &next_h.get(5), operations);
        self.set_bitwise_xor(&next_h_6_tmp, &V14, &next_h.get(6), operations);
        self.set_bitwise_xor(&next_h_7_tmp, &V15, &next_h.get(7), operations);

        for i in 0..8 {
            self.set_to_expression_transition(
                &h.get(i).next(),
                next_h.get(i).expr() * end_bit.not_expr() + iv.get(i).expr() * end_bit.expr(),
            );
        }

        next_h
    }

    fn blake2b_mix(
        &mut self,
        Va: &mut U64Register,
        Vb: &mut U64Register,
        Vc: &mut U64Register,
        Vd: &mut U64Register,
        x: &U64Register,
        y: &U64Register,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: U32Instructions,
    {
        *Va = self.add_u64(Va, Vb, operations);
        *Va = self.add_u64(Va, x, operations);

        *Vd = self.bitwise_xor(Vd, Va, operations);
        *Vd = self.bit_rotate_right(Vd, 32, operations);

        *Vc = self.add_u64(Vc, Vd, operations);

        *Vb = self.bitwise_xor(Vb, Vc, operations);
        *Vb = self.bit_rotate_right(Vb, 24, operations);

        *Va = self.add_u64(Va, Vb, operations);
        *Va = self.add_u64(Va, y, operations);

        *Vd = self.bitwise_xor(Vd, Va, operations);
        *Vd = self.bit_rotate_right(Vd, 16, operations);

        *Vc = self.add_u64(Vc, Vd, operations);

        *Vb = self.bitwise_xor(Vb, Vc, operations);
        *Vb = self.bit_rotate_right(Vb, 63, operations);
    }

    fn permute_msgs<T: Clone>(&mut self, arr: &[T], mix_round_num: usize) -> Vec<T> {
        assert!(mix_round_num <= 12);

        let permutation = SIGMA[mix_round_num % 10];
        let mut result = vec![arr[0].clone(); arr.len()];

        for (from_index, &to_index) in permutation.iter().enumerate() {
            result[to_index] = arr[from_index].clone();
        }

        result
    }
}

impl BLAKE2BGadget {
    pub fn write<F: Field>(
        &self,
        padded_message: &[u8],
        message_len: u64,
        writer: &TraceWriter<F>,
    ) {
        let mut end_bits_values = Vec::new();
        let mut hash_values = Vec::new();
        let mut msg_chunks = Vec::<[F; 8]>::new();

        let num_chunks = padded_message.len() / 128;
        end_bits_values = vec![F::ZERO; num_chunks - 1];
        end_bits_values.push(F::ONE);

        let mut state = INITIAL_HASH;
        let mut bytes_compressed = 0;
        let row_num = 0;
        let t = 0;
        for (chunk_num, chunk) in padded_message.chunks_exact(128).enumerate() {
            let last_chunk = chunk_num == num_chunks - 1;

            if last_chunk {
                bytes_compressed = message_len;
            } else {
                bytes_compressed += 128;
            }

            state = BLAKE2BGadget::compress(
                chunk,
                &mut state,
                bytes_compressed,
                chunk_num == num_chunks - 1,
            );
            hash_values.extend_from_slice(&state.map(u64_to_le_field_bytes::<F>));

            for i in 0..128 / 8 {
                msg_chunks.push(
                    chunk[i * 8..(i + 1) * 8]
                        .iter()
                        .map(|x| F::from_canonical_u8(*x))
                        .collect_vec()
                        .try_into()
                        .unwrap(),
                );
            }

            writer.write_array(
                &self.m,
                &msg_chunks[msg_chunks.len() - 16..msg_chunks.len()],
                chunk_num,
            );
            writer.write(
                &self.t,
                &u64_to_le_field_bytes::<F>(bytes_compressed),
                chunk_num,
            );
            if last_chunk {
                writer.write(&self.last_chunk_bit, &F::ONE, chunk_num);
            }
        }

        writer.write_array(
            &self.initial_hash,
            INITIAL_HASH.map(u64_to_le_field_bytes),
            0,
        );

        writer.write_array(
            &self.initial_hash_compress,
            INITIAL_HASH_COMPRESS.map(u64_to_le_field_bytes),
            0,
        );
        writer.write_array(&self.hash_state, &hash_values, 0);
        writer.write_array(&self.msg_chunks, &msg_chunks, 0);
        writer.write(
            &self.inversion_const,
            &u64_to_le_field_bytes(INVERSION_CONST),
            0,
        );

        (0..512).for_each(|i| {
            writer.write(&self.end_bit, &end_bits_values[i], i * 12 + 11);
            for j in 0..12 {
                let row = i * 12 + j;

                writer.write_array(&self.m, )
            }
        }

    }

    pub fn compress(
        msg_chunk: &[u8],
        state: &mut [u64; 8],
        bytes_compressed: u64,
        last_chunk: bool,
    ) -> [u64; 8] {
        // Set up the work vector V
        let mut V: [u64; 16] = [0; 16];

        V[..8].copy_from_slice(&state[..8]);
        V[8..16].copy_from_slice(&INITIAL_HASH_COMPRESS);

        V[12] ^= bytes_compressed as u64;
        if last_chunk {
            V[14] ^= INVERSION_CONST;
        }

        let msg_u64_chunks = msg_chunk
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect_vec();

        for i in 0..12 {
            let S = SIGMA[i % 10];

            BLAKE2BGadget::mix(
                &mut V,
                0,
                4,
                8,
                12,
                msg_u64_chunks[S[0]],
                msg_u64_chunks[S[1]],
            );
            BLAKE2BGadget::mix(
                &mut V,
                1,
                5,
                9,
                13,
                msg_u64_chunks[S[2]],
                msg_u64_chunks[S[3]],
            );
            BLAKE2BGadget::mix(
                &mut V,
                2,
                6,
                10,
                14,
                msg_u64_chunks[S[4]],
                msg_u64_chunks[S[5]],
            );
            BLAKE2BGadget::mix(
                &mut V,
                3,
                7,
                11,
                15,
                msg_u64_chunks[S[6]],
                msg_u64_chunks[S[7]],
            );

            BLAKE2BGadget::mix(
                &mut V,
                0,
                5,
                10,
                15,
                msg_u64_chunks[S[8]],
                msg_u64_chunks[S[9]],
            );
            BLAKE2BGadget::mix(
                &mut V,
                1,
                6,
                11,
                12,
                msg_u64_chunks[S[10]],
                msg_u64_chunks[S[11]],
            );
            BLAKE2BGadget::mix(
                &mut V,
                2,
                7,
                8,
                13,
                msg_u64_chunks[S[12]],
                msg_u64_chunks[S[13]],
            );
            BLAKE2BGadget::mix(
                &mut V,
                3,
                4,
                9,
                14,
                msg_u64_chunks[S[14]],
                msg_u64_chunks[S[15]],
            );
        }

        for i in 0..8 {
            state[i] ^= V[i];
        }

        for i in 0..8 {
            state[i] ^= V[i + 8];
        }

        *state
    }

    fn mix(V: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
        V[a] = V[a].wrapping_add(V[b]).wrapping_add(x);
        V[d] = (V[d] ^ V[a]).rotate_right(32);
        V[c] = V[c].wrapping_add(V[d]);
        V[b] = (V[b] ^ V[c]).rotate_right(24);
        V[a] = V[a].wrapping_add(V[b]).wrapping_add(y);
        V[d] = (V[d] ^ V[a]).rotate_right(16);
        V[c] = V[c].wrapping_add(V[d]);
        V[b] = (V[b] ^ V[c]).rotate_right(63);
    }

    pub fn pad(msg: &[u8]) -> Vec<u8> {
        if (msg.len() % 128 == 0) && (!msg.is_empty()) {
            msg.to_vec()
        } else {
            let padlen = 128 - (msg.len() % 128);

            let mut padded_msg = Vec::new();
            padded_msg.extend_from_slice(msg);
            padded_msg.extend_from_slice(&vec![0u8; padlen]);
            padded_msg
        }
    }
}

#[cfg(test)]
mod tests {

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::PrimeField64;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Copy)]
    pub struct BLAKE2BTest;

    impl AirParameters for BLAKE2BTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_FREE_COLUMNS: usize = 2032;
        const EXTENDED_COLUMNS: usize = 4722;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_blake2b_stark() {
        type F = GoldilocksField;
        type L = BLAKE2BTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Blake2b test", log::Level::Debug);

        let mut builder = AirBuilder::<L>::new();
        let clk = builder.clock();

        let (mut operations, table) = builder.byte_operations();

        let mut bus = builder.new_bus();
        let channel_idx = bus.new_channel(&mut builder);

        let blake_gadget = builder.process_blake2b(&clk, &mut bus, channel_idx, &mut operations);

        builder.register_byte_lookup(operations, &table);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);
        let writer = generator.new_writer();

        let msg = b"243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917".to_vec();
        let digest = "369ffcc61c51d8ed04bf30a9e8cf79f8994784d1e3f90f32c3182e67873a3238";

        let padded_msg = BLAKE2BGadget::pad(&msg).into_iter().collect::<Vec<_>>();

        let expected_digest = hex::decode(digest).unwrap().into_iter().collect::<Vec<_>>();

        timed!(timing, "Write the execusion trace", {
            table.write_table_entries(&writer);
            blake_gadget.write(&padded_msg, msg.len() as u64, &writer);
            for i in 0..L::num_rows() {
                writer.write_row_instructions(&generator.air_data, i);
                let end_bit = writer.read(&blake_gadget.end_bit, i);
                if end_bit == F::ONE {
                    let hash: [[GoldilocksField; 8]; 4] =
                        writer.read_array(&blake_gadget.h.get_subarray(0..4), i);
                    let calculated_hash_bytes = hash
                        .iter()
                        .flatten()
                        .map(|x| x.to_canonical_u64() as u8)
                        .collect_vec();
                    assert_eq!(calculated_hash_bytes, expected_digest);
                }
            }
            table.write_multiplicities(&writer);
        });

        let public_inputs = writer.0.public.read().unwrap().clone();
        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        timed!(
            timing,
            "Stark proof and verify",
            test_starky(&stark, &config, &generator, &public_inputs)
        );

        // // Generate recursive proof
        // timed!(
        //     timing,
        //     "Recursive proof generation and verification",
        //     test_recursive_starky(stark, config, generator, &public_inputs)
        // );

        timing.print();
    }
}
