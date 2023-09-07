use crate::chip::bool::SelectInstruction;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::uint::bytes::gadget::operation;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::{ByteArrayRegister, U64Register};
use crate::chip::AirParameters;

pub type U64Value<T> = <U64Register as Register>::Value<T>;

#[derive(Debug, Clone)]
pub struct Blake2bGadget {
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

impl<L: AirParameters> AirBuilder<L> {
    fn blake2b_compress(
        &mut self,
        clk: &ElementRegister,
        m: &ArrayRegister<U64Register>,
        h: &ArrayRegister<U64Register>,
        iv: &ArrayRegister<U64Register>,
        inversion_const: &U64Register,
        input_chunk: ArrayRegister<U64Register>,
        t: U64Register, // assumes t is not more than u64
        last_block_bit: &BitRegister,
        operations: &mut ByteLookupOperations,
    ) -> ArrayRegister<U64Register>
    where
        L::Instruction: U32Instructions,
        L::Instruction: From<SelectInstruction<ByteArrayRegister<8>>>,
    {
        // Initialize local work vector V
        let V0 = h.get(0);
        let V1 = h.get(1);
        let V2 = h.get(2);
        let V3 = h.get(3);
        let V4 = h.get(4);
        let V5 = h.get(5);
        let V6 = h.get(6);
        let V7 = h.get(7);

        let V8 = iv.get(8);
        let V9 = iv.get(9);
        let V10 = iv.get(10);
        let V11 = iv.get(11);
        let V12 = iv.get(12);
        let V13 = iv.get(13);
        let V14 = iv.get(14);
        let V15 = iv.get(15);

        for i in 0..32 {
            self.set_to_expression_first_row(&h.get(i), iv.get(i).expr());
        }

        V12 = self.bitwise_xor(&V12, &t, operations);

        // We assume that t is no more than u64, so we don't modify V13

        self.select::<ByteArrayRegister<8>>(
            last_block_bit,
            &self.bitwise_xor(&V14, inversion_const, operations),
            &V14,
        );

        for i in 0..12 {
            self.blake2b_mix(
                &V0,
                &V4,
                &V8,
                &V12,
                &m.get(SIGMA[i % 10][0]),
                &m.get(SIGMA[i % 10][1]),
                operations,
            );

            self.blake2b_mix(
                &V1,
                &V5,
                &V9,
                &V13,
                &m.get(SIGMA[i % 10][2]),
                &m.get(SIGMA[i % 10][3]),
                operations,
            );

            self.blake2b_mix(
                &V2,
                &V6,
                &V10,
                &V14,
                &m.get(SIGMA[i % 10][4]),
                &m.get(SIGMA[i % 10][5]),
                operations,
            );

            self.blake2b_mix(
                &V3,
                &V7,
                &V11,
                &V15,
                &m.get(SIGMA[i % 10][6]),
                &m.get(SIGMA[i % 10][7]),
                operations,
            );

            self.blake2b_mix(
                &V0,
                &V5,
                &V10,
                &V15,
                &m.get(SIGMA[i % 10][8]),
                &m.get(SIGMA[i % 10][9]),
                operations,
            );

            self.blake2b_mix(
                &V1,
                &V6,
                &V11,
                &V12,
                &m.get(SIGMA[i % 10][10]),
                &m.get(SIGMA[i % 10][11]),
                operations,
            );

            self.blake2b_mix(
                &V2,
                &V7,
                &V8,
                &V13,
                &m.get(SIGMA[i % 10][12]),
                &m.get(SIGMA[i % 10][13]),
                operations,
            );

            self.blake2b_mix(
                &V3,
                &V4,
                &V9,
                &V14,
                &m.get(SIGMA[i % 10][14]),
                &m.get(SIGMA[i % 10][15]),
                operations,
            );
        }

        let next_h = self.alloc_array::<U64Register>(8);
        next_h.set(0, self.bitwise_xor(&h.get(0), &V0, operations));
    }

    fn blake2b_mix(
        &self,
        mut Va: &U64Register,
        mut Vb: &U64Register,
        mut Vc: &U64Register,
        mut Vd: &U64Register,
        x: &U64Register,
        y: &U64Register,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: U32Instructions,
    {
        Va = &self.add_u64(
            &self.add_u64(&Va, &Vb, &mut &operations),
            &x,
            &mut &operations,
        );

        Vd = &self.bit_rotate_right(
            &self.bitwise_xor(&Vd, &Va, &mut &operations),
            32,
            &mut &operations,
        );

        Vc = &self.add_u64(&Vc, &Vd, &mut &operations);

        Vb = &self.bit_rotate_right(
            &self.bitwise_xor(&Vb, &Vc, &mut &operations),
            24,
            &mut &operations,
        );

        Va = &self.add_u64(
            &self.add_u64(&Va, &Vb, &mut &operations),
            &y,
            &mut &operations,
        );

        Vd = &self.bit_rotate_right(
            &self.bitwise_xor(&Vd, &Va, &mut &operations),
            16,
            &mut &operations,
        );

        Vc = &self.add_u64(&Vc, &Vd, &mut &operations);

        Vb = &self.bit_rotate_right(
            &self.bitwise_xor(&Vb, &Vc, &mut &operations),
            63,
            &mut &operations,
        );
    }
}
