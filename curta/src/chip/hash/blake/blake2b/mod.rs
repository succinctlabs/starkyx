pub mod builder_gadget;
pub mod generator;

use itertools::Itertools;

use crate::chip::bool::SelectInstruction;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::{ByteArrayRegister, U64Register};
use crate::chip::AirParameters;

pub type U64Value<T> = <U64Register as Register>::Value<T>;

#[derive(Debug, Clone)]
pub struct BLAKE2BGadget {
    /// The input chunks processed into 16-words of U64 values
    pub public_word: ArrayRegister<U64Register>,
    /// The hash states at all 1024 rounds
    pub state: ArrayRegister<U64Register>,
    /// Signifies when to reset the state to the initial hash
    pub end_bit: BitRegister,

    pub(crate) end_bits_public: ArrayRegister<BitRegister>,
    pub(crate) initial_state: ArrayRegister<U64Register>,
}

#[derive(Debug, Clone)]
pub struct Blake2bPublicData<T> {
    pub public_w: Vec<U64Value<T>>,
    pub hash_state: Vec<U64Value<T>>,
    pub end_bits: Vec<T>,
}

// Note that for this blake2b implementation, we don't support a key input and
// we assume that the output is 32 bytes
// So that means the initial hash entry to be
// 0x6a09e667f3bcc908 xor 0x01010020
const INITIAL_HASH: [u64; 16] = [
    0x6a09e667f2bdc928,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
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

#[allow(clippy::too_many_arguments)]
impl<L: AirParameters> AirBuilder<L> {
    fn blake2b_compress(
        &mut self,
        clk: &ElementRegister,
        m: &ArrayRegister<U64Register>,
        h: &ArrayRegister<U64Register>,
        iv: &ArrayRegister<U64Register>,
        inversion_const: &U64Register,
        t: U64Register, // assumes t is not more than u64
        last_block_bit: &BitRegister,
        operations: &mut ByteLookupOperations,
        end_bit: &BitRegister,
    ) -> ArrayRegister<U64Register>
    where
        L::Instruction: U32Instructions,
        L::Instruction: From<SelectInstruction<ByteArrayRegister<8>>>,
    {
        for i in 0..32 {
            self.set_to_expression_first_row(&h.get(i), iv.get(i).expr());
        }

        let mut V0 = h.get(0);
        let mut V1 = h.get(1);
        let mut V2 = h.get(2);
        let mut V3 = h.get(3);
        let mut V4 = h.get(4);
        let mut V5 = h.get(5);
        let mut V6 = h.get(6);
        let mut V7 = h.get(7);

        let mut V8 = iv.get(8);
        let mut V9 = iv.get(9);
        let mut V10 = iv.get(10);
        let mut V11 = iv.get(11);
        let mut V12 = iv.get(12);
        let mut V13 = iv.get(13);
        let mut V14 = iv.get(14);
        let mut V15 = iv.get(15);

        V12 = self.bitwise_xor(&V12, &t, operations);
        // We assume that t is no more than u64, so we don't modify V13

        let tmp = self.bitwise_xor(&V14, inversion_const, operations);
        self.select::<ByteArrayRegister<8>>(last_block_bit, &tmp, &V14);

        for i in 0..12 {
            self.blake2b_mix(
                &mut V0,
                &mut V4,
                &mut V8,
                &mut V12,
                &m.get(SIGMA[i % 10][0]),
                &m.get(SIGMA[i % 10][1]),
                operations,
            );

            self.blake2b_mix(
                &mut V1,
                &mut V5,
                &mut V9,
                &mut V13,
                &m.get(SIGMA[i % 10][2]),
                &m.get(SIGMA[i % 10][3]),
                operations,
            );

            self.blake2b_mix(
                &mut V2,
                &mut V6,
                &mut V10,
                &mut V14,
                &m.get(SIGMA[i % 10][4]),
                &m.get(SIGMA[i % 10][5]),
                operations,
            );

            self.blake2b_mix(
                &mut V3,
                &mut V7,
                &mut V11,
                &mut V15,
                &m.get(SIGMA[i % 10][6]),
                &m.get(SIGMA[i % 10][7]),
                operations,
            );

            self.blake2b_mix(
                &mut V0,
                &mut V5,
                &mut V10,
                &mut V15,
                &m.get(SIGMA[i % 10][8]),
                &m.get(SIGMA[i % 10][9]),
                operations,
            );

            self.blake2b_mix(
                &mut V1,
                &mut V6,
                &mut V11,
                &mut V12,
                &m.get(SIGMA[i % 10][10]),
                &m.get(SIGMA[i % 10][11]),
                operations,
            );

            self.blake2b_mix(
                &mut V2,
                &mut V7,
                &mut V8,
                &mut V13,
                &m.get(SIGMA[i % 10][12]),
                &m.get(SIGMA[i % 10][13]),
                operations,
            );

            self.blake2b_mix(
                &mut V3,
                &mut V4,
                &mut V9,
                &mut V14,
                &m.get(SIGMA[i % 10][14]),
                &m.get(SIGMA[i % 10][15]),
                operations,
            );
        }

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
}

impl BLAKE2BGadget {
    pub fn compress(
        msg_chunk: &[u8; 256],
        state: &mut [u64; 16],
        bytes_compressed: usize,
        last_chunk: bool,
    ) -> [u64; 16] {
        // Set up the work vector V
        let mut V: [u64; 16] = [0; 16];

        V[..8].copy_from_slice(&state[..8]);
        V[8..16].copy_from_slice(&INITIAL_HASH[..8]);

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
        let padlen = 128 - (msg.len() % 128);

        let mut padded_msg = Vec::new();
        padded_msg.extend_from_slice(msg);
        padded_msg.extend_from_slice(&vec![0u8; padlen]);
        padded_msg
    }
}
