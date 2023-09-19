use core::borrow::Borrow;

use plonky2::field::types::Field;

use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::{ByteArrayRegister, U64Register};
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::AirParameters;

pub struct Keccak256Gadget {
    pub state: ArrayRegister<ByteArrayRegister<8>>,
    pub(crate) round_constant: U64Register,
    cycle_24_end_bit: BitRegister,
    msg_end_bit: BitRegister,
    msg_block: ArrayRegister<ByteArrayRegister<8>>,
}

#[derive(Debug, Clone)]
pub struct Keccak256PublicData {
    // pub public_w: Vec<U32Value<T>>,
    // pub hash_state: Vec<U32Value<T>>,
    // pub end_bits: Vec<T>,
}

#[rustfmt::skip]
pub const KECCAKF_ROTC: [[u32; 5]; 5] = [
    [0, 1, 62, 28, 27],
    [36, 44, 6, 55, 20],
    [3, 10, 43, 25, 39],
    [41, 45, 15, 21, 8],
    [18, 2, 61, 56, 14]
];

#[rustfmt::skip]
pub const KECCAKF_RNDC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
];

// impl constraint for keccack256
// state register
// todo: figure out round_constant and rotation offset
impl<L: AirParameters> AirBuilder<L> {
    // trace table layout for keccak_f
    //      state  |  state_after_theta | state_after_rhopi | state_after_chi | round_constant
    // i    input        x                       x                x            RC[i]
    // i+1  new_state
    pub fn keccak_f(&mut self, operations: &mut ByteLookupOperations) -> Keccak256Gadget
    where
        L::Instruction: U32Instructions,
    {
        // alloate 5*5 u64 register for storing 25 8byte lanes
        // state is 25 * 64 bit
        // a[i,j] is (j * 5 + i)th lane
        //    i
        // xxxxx
        // xxxxx
        // xxxxx
        // xxxxx  j  3*5 + 3 = 18th lane
        // xxxxx
        // i == x axis, refer to column, j == y axis, refer to row, this diff from most competitve programmer's convention, but follow the keccak spec, so pls tolerate. so bit weird but a[i,j] actually refer to j * 5 + i NOT i * 5 + j.
        // alloc round_const, write it later outside circuit, constrain later
        let round_constant = self.alloc::<U64Register>();
        // 00 .. 01 | 00..01 | .. every 24 rows, inform whether to process next block or within rounds of the same block
        let cycle_24_end_bit = self.alloc::<BitRegister>();
        // inform whether a msg ended so the hash value need to be reset (this useful for batching multiple msg computation to one table)
        let msg_end_bit = self.alloc::<BitRegister>();
        // msg blocks in 1088 = 64 * 17 bit, should be same within each block (24 round), need constrain
        // optimization trick if use 25 and padd eveyrthing with 0 since c bits are all 0, so can directly use msg_block as the initial state instead of do another column
        let msg_block = self.alloc_array::<U64Register>(25);

        // 00 .. 00 | 00..00 |
        // unsure this is correct..
        // how to constrain it to come constant, i.e. the fixed column for the circuit?
        // for i in 0..24 {
        //     // self.assert_equal(&round_const.get(i)., &ByteArrayRegister::from(KECCAKF_RNDC[i]));
        // }

        // maybe need to boundary the initial state?
        // define boundary constraint for theta
        let state = self.alloc_array::<U64Register>(25);

        // constrain the first row's state is equal to msg_block since it's not included in the transition constraint
        for x in 0..5 {
            for y in 0..5 {
                self.assert_equal_first_row(&state.get(x + y * 5), &msg_block.get(x + y * 5));
            }
        }

        // theta
        // how to constrain an array value follows a transition constraint in for loop?
        let c_arr = self.alloc_array::<U64Register>(5);
        for x in 0..5 {
            let mut c_i = state.get(x);
            for y in 1..5 {
                c_i = self.bitwise_xor(&c_i, &state.get(y * 5 + x), operations);
            }
            // Does it suffice to constrain for every row of c_i, it need to satisfy its relationship with state at the same row?
            self.set_to_expression(&c_arr.get(x), c_i.expr());
        }

        let d_arr = self.alloc_array::<U64Register>(5);
        // initial state doesn't exist? or don't want to constrain? first row of it is empty?
        let state_after_theta = self.alloc_array::<U64Register>(25);

        for x in 0..5 {
            let temp = self.bit_rotate_right(&c_arr.get((x + 1) % 5), 1, operations);
            let d_i = self.bitwise_xor(&c_arr.get((x + 4) % 5), &temp, operations);
            self.set_to_expression(&d_arr.get(x), d_i.expr());
            for y in 0..5 {
                // make sure state_after_theta follows the theta transition of state
                let tmp = self.bitwise_xor(&state.get(y * 5 + x), &d_i, operations);
                self.set_to_expression(&(state_after_theta.get(y * 5 + x)), tmp.expr());
            }
        }

        let state_after_rhopi = self.alloc_array::<U64Register>(25);
        // 0,0 has no change, direct copy constraint
        self.set_to_expression(&state_after_rhopi.get(0), state_after_theta.get(0).expr());

        // rho and pi
        for x in 0..5 {
            for y in 0..5 {
                // x is column, y is row
                // y, 2x+3y, is pi_idx in the flatten version
                if x + y != 0 {
                    let pi_idx = ((2 * x + 3 * y) % 5) * 5 + y;
                    let tmp = self.bit_rotate_right(
                        &state_after_theta.get(y * 5 + x),
                        KECCAKF_ROTC[y][x].try_into().unwrap(),
                        operations,
                    );
                    self.set_to_expression(&state_after_rhopi.get(pi_idx), tmp.expr());
                }
            }
        }

        let state_after_chi = self.alloc_array::<U64Register>(25);
        // chi
        for x in 0..5 {
            for y in 0..5 {
                let tmp1 =
                    self.bitwise_not(&state_after_rhopi.get((x + 1) % 5 + y * 5), operations);
                let tmp2 = self.bitwise_and(
                    &tmp1,
                    &state_after_rhopi.get((x + 2) % 5 + y * 5),
                    operations,
                );
                let tmp3 = self.bitwise_xor(&state_after_rhopi.get(x + y * 5), &tmp2, operations);
                self.set_to_expression(&state_after_chi.get(x + y * 5), tmp3.expr());
            }
        }

        let state_after_iota = self.alloc_array::<U64Register>(25);
        // iota
        let tmp = self.bitwise_xor(&state_after_chi.get(0), &round_constant, operations);
        self.set_to_expression(&state_after_iota.get(0), tmp.expr());

        // constrain the other val of next state is the same as chi's result
        for x in 0..5 {
            for y in 0..5 {
                // can if used here? if not, guess need to pass some indicator bit to help identity 0,0
                if x + y > 0 {
                    self.set_to_expression(
                        &state_after_iota.get(x + y * 5),
                        state_after_chi.get(x + y * 5).expr(),
                    );
                }
            }
        }

        let state_after_xor = self.alloc_array::<U64Register>(25);

        // set state_after_xor, the state as input for next round of keccak_f if there's still msg block to be processed
        // msg_block.next() is the new msg block for next round, position in row 24 (0 idxed)
        // currently this state_after_xor is still on the same row with each round since
        for x in 0..5 {
            for y in 0..5 {
                let idx = x + y * 5;
                if idx < 17 {
                    let tmp = self.bitwise_xor(
                        &state_after_iota.get(idx),
                        &msg_block.get(idx),
                        operations,
                    );
                    self.set_to_expression(&state_after_xor.get(idx), tmp.expr());
                } else {
                    // c bits are directly copied over since it's never changed.
                    self.set_to_expression(
                        &state_after_xor.get(idx),
                        state_after_iota.get(idx).expr(),
                    );
                }
            }
        }

        // set next state input to keccak_f
        // if cycle not end, i.e cycle_bit = 0 (within same block)
        // set to state_after_iota
        // else
        //    if still have msg block to process, i.e msg_bit = 0
        //        set to state_after_xor
        //    else
        //        set to initial state, i.e. msg_block's value (a xor 0 = a and c bit are all zero)
        for x in 0..5 {
            for y in 0..5 {
                let idx = x + y * 5;
                self.set_to_expression_transition(
                    &state.get(idx).next(),
                    state_after_iota.get(idx).expr() * cycle_24_end_bit.not_expr()
                        + (state_after_xor.get(idx).expr() * msg_end_bit.not_expr()
                            + msg_block.get(idx).expr() * msg_end_bit.expr())
                            * cycle_24_end_bit.expr(),
                );
            }
        }

        Keccak256Gadget {
            state,
            round_constant,
            cycle_24_end_bit,
            msg_end_bit,
            msg_block,
        }
    }
}

impl Keccak256Gadget {
    pub fn generate_trace<F: Field, I: IntoIterator>(
        &self,
        _padded_messages: I,
        _writer: &TraceWriter<F>,
    ) -> Keccak256PublicData
    where
        I::Item: Borrow<[u8]>,
    {
        // TODO: generate the trace so it can satisfy the constraint,
        // note it's tricky since there're many rows where only one column (set of column) is changing, all the other column including state register are basically directly copying to next row, need to carefully compute the gap in the table and accurately put the right value in right row (position), is it correct?....
        // the other cleaner method would be instead of expand to other register(column), use same state register but is the .next() in-place, if not, how to constrain multiple rows, use .next().next() ..? is it clean?
        // currently there's indeed too many columns, which is bad...for both correct trace generation implementation and performance (proof generation time) i think..
        // also figure out how to pad and discard for variable length input, discard may need some bit array tricks.
        todo!()
    }

    // pad for keccak256, msg is array of bytes, but return it's series of bits, i.e. 0 and 1, but just represented as array of u8.
    // if this confusing may change to some other name and structure
    pub fn pad(msg: &[u8]) -> Vec<u8> {
        const R: usize = 1088;
        let mut bits = Self::into_bits(msg);
        bits.push(1);
        while (bits.len() + 1) % R != 0 {
            bits.push(0);
        }
        bits.push(1);
        bits
    }

    /// Converts bytes into bits LE
    pub fn into_bits(bytes: &[u8]) -> Vec<u8> {
        let mut bits: Vec<u8> = vec![0; bytes.len() * 8];
        for (byte_idx, byte) in bytes.iter().enumerate() {
            for idx in 0u64..8 {
                bits[byte_idx * 8 + (idx as usize)] = (*byte >> idx) & 1;
            }
        }
        bits
    }

    // convert bits into bytes LE
    pub fn into_bytes(bits: &[u8]) -> Vec<u8> {
        debug_assert!(bits.len() % 8 == 0, "bits not a multiple of 8");
        let mut bytes = Vec::new();
        for byte_bits in bits.chunks(8) {
            let mut value = 0u8;
            for (idx, bit) in byte_bits.iter().enumerate() {
                value += *bit << idx;
            }
            bytes.push(value);
        }
        bytes
    }

    // todo: change to multiple msg batched together
    // todo: use const to make code more readable
    // msg is array of bytes
    pub fn write_msg_block<F: Field, L: AirParameters>(&self, msg: &[u8], writer: &TraceWriter<F>) {
        let padded_msg = Keccak256Gadget::pad(msg);
        debug_assert!(
            padded_msg.len() % 1088 == 0,
            "padded msg is not multiple of 1088 bits"
        );
        // 17 u64 + 8 u64
        let num_block_in_msg = padded_msg.len() / 1088;
        let mut writable_msgs = vec![];
        for msg_block in padded_msg.chunks(1088) {
            println!("{:?}", msg_block.len());
            // pad the msg_block with 0 c_bits till reach 1600 = 200 * 8 bits
            let mut msg_block_with_c_bits = vec![0u8; 1600];
            msg_block_with_c_bits[..1088].copy_from_slice(msg_block);
            let bytes = Self::into_bytes(&msg_block_with_c_bits);
            debug_assert!(
                bytes.len() == 200,
                "each msg block after c capabity bits padding should have exactly 200 bytes"
            );
            // group bit array to u64 for writing purpose
            let group_in_u64 = bytes
                .chunks_exact(8)
                .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>();
            debug_assert!(group_in_u64.len() == 25, "should grouped into exactly 25 lanes for msg that'll also used as initial state for each fresh computation");
            let mut r: Vec<[F; 8]> = vec![];
            for val in group_in_u64 {
                r.push(u64_to_le_field_bytes(val));
            }
            writable_msgs.push(r);
        }

        debug_assert!(writable_msgs.len() == num_block_in_msg, "for each new block, there's a new msg chuck that needs to be written to the msg_block column, and then repeat to fill all the table");

        let msg_end_gap = num_block_in_msg * 24;
        for i in 0..L::num_rows() {
            // write msg_block
            if (i + 1) % 24 == 0 {
                writer.write_array(
                    &self.msg_block,
                    writable_msgs[i % num_block_in_msg].clone(),
                    i,
                );
            }

            // write the very first row, bit special due to constraint and table layout..
            if i == 0 {
                writer.write_array(&self.msg_block, writable_msgs[0].clone(), 0);
                writer.write_array(&self.state, writable_msgs[0].clone(), 0);
            }

            // write msg_end_bit
            if (i + 1) % msg_end_gap == 0 {
                writer.write(&self.msg_end_bit, &F::ONE, i);
            } else {
                writer.write(&self.msg_end_bit, &F::ZERO, i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct Keccak256Test;

    impl AirParameters for Keccak256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_FREE_COLUMNS: usize = 3431;
        const EXTENDED_COLUMNS: usize = 6234;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_pad() {
        let msg = "abc".as_bytes();
        let res = Keccak256Gadget::pad(msg);
        assert_eq!(res.len() % 136, 0);
    }

    #[test]
    fn test_keccak_256_stark() {
        type F = GoldilocksField;
        type L = Keccak256Test;
        type SC = PoseidonGoldilocksStarkConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut builder = AirBuilder::<L>::new();

        let (mut operations, table) = builder.byte_operations();

        let keccak_gadget = builder.keccak_f(&mut operations);

        builder.register_byte_lookup(operations, &table);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);
        let writer = generator.new_writer();
        table.write_table_entries(&writer);

        // write the initial value of state to 0th row
        // for i in 0..25 {
        //     writer.write(&keccak_f_gadget.state.get(i), &u64_to_le_field_bytes(0), 0);
        // }
        // write the intial a
        // writer.write(&keccak_f_gadget.a, &u64_to_le_field_bytes(5), 0);
        // println!("{}", L::num_rows());
        println!("{}", generator.air_data.instructions.len());
        keccak_gadget.write_msg_block::<F, L>("abc".as_bytes(), &writer);

        // write the initial input to state to be
        // new state would be written to state.next() through instructions given by set_to_expression_transition
        // for i in 0..25 {
        //     writer.write(&keccak_gadget.state.get(i), &u64_to_le_field_bytes(0), i);
        // }

        for i in 0..L::num_rows() {
            let round_constant_value = u64_to_le_field_bytes::<F>(KECCAKF_RNDC[i % 24]);
            writer.write(&keccak_gadget.round_constant, &round_constant_value, i);

            if (i + 1) % 24 == 0 {
                writer.write(&keccak_gadget.cycle_24_end_bit, &F::ONE, i);
            } else {
                writer.write(&keccak_gadget.cycle_24_end_bit, &F::ZERO, i);
            }

            writer.write_row_instructions(&generator.air_data, i);
        }

        table.write_multiplicities(&writer);

        // after one round result should be at row_index = 1

        // for i in 0..L::num_rows() {
        //     println!("{:?}", writer.read(&keccak_gadget.round_constant, i));
        // }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        test_starky(&stark, &config, &generator, &[]);
    }
}

// todo
// offset in register, and constrain it for each row?
// readablity
// opt
//
