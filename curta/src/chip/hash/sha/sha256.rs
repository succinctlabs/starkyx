use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::lookup_table::table::ByteLookupTable;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::{ByteArrayRegister, U32Register};
use crate::chip::AirParameters;
use crate::math::prelude::*;

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INITIAL_HASH: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub fn round_constants<F: Field>() -> [[F; 4]; 64] {
    ROUND_CONSTANTS
        .map(u32::to_le_bytes)
        .map(|x| x.map(F::from_canonical_u8))
}

pub fn first_hash_value<F: Field>() -> [[F; 4]; 8] {
    [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ]
    .map(u32::to_le_bytes)
    .map(|x| x.map(F::from_canonical_u8))
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl<L: AirParameters> AirBuilder<L> {
    pub fn sha_premessage(&mut self, 
        w : &U32Register,
        w_minus_2 : &U32Register,
        w_minus_7 : &U32Register,
        w_minus_15 : &U32Register,
        w_minus_16 : &U32Register,
        operations: &mut ByteLookupOperations)
    where
        L::Instruction: U32Instructions,
    {

    }

    pub fn sha_256_step(
        &mut self,
        hash: &ArrayRegister<U32Register>,
        hash_bit : &BitRegister,
        msg: &ArrayRegister<U32Register>,
        w: &U32Register,
        round_constant: &U32Register,
        operations: &mut ByteLookupOperations,
    ) -> ArrayRegister<U32Register>
    where
        L::Instruction: U32Instructions,
    {
        let cycle_64 = self.cycle(6);

        // The sha round

        // Initialize working variables
        let a = msg.get(0);
        let b = msg.get(1);
        let c = msg.get(2);
        let d = msg.get(3);
        let e = msg.get(4);
        let f = msg.get(5);
        let g = msg.get(6);
        let h = msg.get(7);

        // Calculate sum_1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let e_rotate_6 = self.bit_rotate_right(&e, 6, operations);
        let e_rotate_11 = self.bit_rotate_right(&e, 11, operations);
        let e_rotate_25 = self.bit_rotate_right(&e, 25, operations);
        let mut sum_1 = self.bitwise_xor(&e_rotate_6, &e_rotate_11, operations);
        sum_1 = self.bitwise_xor(&sum_1, &e_rotate_25, operations);

        // Calculate ch = (e & f) ^ (!e & g);
        let e_and_f = self.bitwise_and(&e, &f, operations);
        let not_e = self.bitwise_not(&e, operations);
        let not_e_and_g = self.bitwise_and(&not_e, &g, operations);
        let ch = self.bitwise_xor(&e_and_f, &not_e_and_g, operations);

        // Calculate temp_1 = h + sum_1 +ch + round_constant + w;
        let mut temp_1 = self.add_u32(&h, &sum_1, operations);
        temp_1 = self.add_u32(&temp_1, &ch, operations);
        temp_1 = self.add_u32(&temp_1, &round_constant, operations);
        temp_1 = self.add_u32(&temp_1, &w, operations);

        // Calculate sum_0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let a_rotate_2 = self.bit_rotate_right(&a, 2, operations);
        let a_rotate_13 = self.bit_rotate_right(&a, 13, operations);
        let a_rotate_22 = self.bit_rotate_right(&a, 22, operations);
        let mut sum_0 = self.bitwise_xor(&a_rotate_2, &a_rotate_13, operations);
        sum_0 = self.bitwise_xor(&sum_0, &a_rotate_22, operations);

        // Calculate maj = (a & b) ^ (a & c) ^ (b & c);
        let a_and_b = self.bitwise_and(&a, &b, operations);
        let a_and_c = self.bitwise_and(&a, &c, operations);
        let b_and_c = self.bitwise_and(&b, &c, operations);
        let mut maj = self.bitwise_xor(&a_and_b, &a_and_c, operations);
        maj = self.bitwise_xor(&maj, &b_and_c, operations);

        // Calculate temp_2 = sum_0 + maj;
        let temp_2 = self.add_u32(&sum_0, &maj, operations);

        // Calculate the next cycle values
        let a_next = self.add_u32(&temp_1, &temp_2, operations);
        let b_next = a;
        let c_next = b;
        let d_next = c;
        let e_next = self.add_u32(&d, &temp_1, operations);
        let f_next = e;
        let g_next = f;
        let h_next = g;

        let msg_next = [
            a_next, b_next, c_next, d_next, e_next, f_next, g_next, h_next,
        ];

        // Assign next values to the next row registers based on the cycle bit
        let bit = cycle_64.end_bit;
        for i in 0..8 {
            self.set_to_expression_transition(
                &msg.get(i).next(),
                msg_next[i].expr() * bit.not_expr() + hash.get(i).next().expr() * bit.expr(),
            );
        }

        // Assign the hash values in the end of the round
        let hash_next = self.alloc_array::<U32Register>(8);
        for i in 0..8 {
            self.set_add_u32(&hash.get(i), &msg_next[i], &hash_next.get(i), operations);
        }

        hash_next 
    }
}

#[cfg(test)]
mod tests {

    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::register::RegisterSized;
    use crate::chip::uint::bytes::operations::value::ByteOperation;
    use crate::chip::uint::bytes::register::ByteRegister;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::uint::register::ByteArrayRegister;
    use crate::chip::AirParameters;
    use crate::math::prelude::*;
    use crate::plonky2::field::Field;

    #[derive(Debug, Clone, Copy)]
    pub struct SHA256Test;

    impl const AirParameters for SHA256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_FREE_COLUMNS: usize = 512;
        const EXTENDED_COLUMNS: usize = 945;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    fn sha_256_pre(chunk: [u32; 16]) -> [u32; 64] {
        let mut w = [0u32; 64];

        for i in 0..16 {
            w[i] = chunk[i];
        }

        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        w
    }

    fn sha_round(hash: [u32; 8], w: &[u32], round_constants: [u32; 64]) -> [u32; 8] {
        let mut msg = hash;
        for i in 0..64 {
            msg = sha_step(msg, w[i], round_constants[i]);
        }

        [
            hash[0].wrapping_add(msg[0]),
            hash[1].wrapping_add(msg[1]),
            hash[2].wrapping_add(msg[2]),
            hash[3].wrapping_add(msg[3]),
            hash[4].wrapping_add(msg[4]),
            hash[5].wrapping_add(msg[5]),
            hash[6].wrapping_add(msg[6]),
            hash[7].wrapping_add(msg[7]),
        ]
    }

    fn sha_step(msg: [u32; 8], w_i: u32, round_constant: u32) -> [u32; 8] {
        let mut a = msg[0];
        let mut b = msg[1];
        let mut c = msg[2];
        let mut d = msg[3];
        let mut e = msg[4];
        let mut f = msg[5];
        let mut g = msg[6];
        let mut h = msg[7];

        let sum_1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp_1 = h
            .wrapping_add(sum_1)
            .wrapping_add(ch)
            .wrapping_add(round_constant)
            .wrapping_add(w_i);
        let sum_0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp_2 = sum_0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp_1);
        d = c;
        c = b;
        b = a;
        a = temp_1.wrapping_add(temp_2);

        [a, b, c, d, e, f, g, h]
    }

    fn sha_256_process(chunk: [u32; 16], hash: [u32; 8], round_constants: [u32; 64]) -> [u32; 8] {
        let mut w = sha_256_pre(chunk);
        sha_round(hash, &w, round_constants)
    }

    #[test]
    fn test_sha_256_stark() {
        type F = GoldilocksField;
        type L = SHA256Test;
        type SC = PoseidonGoldilocksStarkConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Sha256 test", log::Level::Debug);

        let mut builder = AirBuilder::<L>::new();

        let (mut operations, mut table) = builder.byte_operations();

        let w = builder.alloc::<U32Register>();
        let msg_array = builder.alloc_array::<U32Register>(8);
        let hash = builder.alloc_array::<U32Register>(8);
        let round_constant = builder.alloc::<U32Register>();
        let hash_bit = builder.alloc::<BitRegister>();

        let hash_next =
            builder.sha_256_step(&hash, &hash_bit, &msg_array, &w, &round_constant, &mut operations);

        // let dummy = builder.alloc::<ByteRegister>();
        // let dummy_range = ByteOperation::Range(dummy);
        // builder.set_byte_operation(&dummy_range, &mut operations);

        builder.register_byte_lookup(operations, &mut table);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        let empty_msg = b"";
        let expected_digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        let mut padded_msg = [0u32; 16];
        padded_msg[0] = 1 << 31;

        let initial_hash = INITIAL_HASH;
        let round_constants = ROUND_CONSTANTS;

        let to_field = |x: u32| x.to_le_bytes().map(F::from_canonical_u8);
        let to_val = |arr: [F; 4]| u32::from_le_bytes(arr.map(|x| x.as_canonical_u64() as u8));

        let w_val = sha_256_pre(padded_msg);

        table.write_table_entries(&writer);
        for i in 0..1024 {
            writer.write_array(&msg_array, INITIAL_HASH.map(to_field), i * 64);
            writer.write_array(&hash, INITIAL_HASH.map(to_field), i * 64);
            writer.write_array(&hash, INITIAL_HASH.map(to_field), i * 64 + 63); 
            for j in 0..64 {
                let row = i * 64 + j;
                writer.write(&round_constant, &to_field(round_constants[j]), row);
                writer.write(&w, &to_field(w_val[j]), row);

                writer.write_row_instructions(&air, row);
            }
        }
        table.write_multiplicities(&writer);

        let msg_val = |i| writer.read_array::<_, 8>(&msg_array, i).map(to_val);
        let hash_next_val = |i| writer.read_array::<_, 8>(&hash_next, i).map(to_val);

        let exp_message = |i| {
            let mut msg = initial_hash;
            for j in 0..i {
                msg = sha_step(msg, w_val[j], round_constants[j]);
            }
            msg
        };
        assert_eq!(msg_val(0), initial_hash);
        // assert_eq!(msg_val(1), hash_next_val(0));

        // assert_eq!(exp_message(1), hash_next_val(0));

        // assert_eq!(hash_next_val(63), exp_message(64));
        let hash_val = sha_round(initial_hash, &w_val, round_constants);
        assert_eq!(hash_next_val(63), hash_val);

        let expected_hash: [u32; 8] = hex::decode(expected_digest)
            .unwrap()
            .chunks_exact(4)
            .map(|x| u32::from_be_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        assert_eq!(hash_val, expected_hash);

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        timing.print();
    }
}
