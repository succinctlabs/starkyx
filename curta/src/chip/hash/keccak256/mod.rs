use core::borrow::Borrow;

use plonky2::field::types::Field;

use crate::chip::{builder::{AirBuilder}, AirParameters, trace::writer::TraceWriter, uint::{register::{U64Register, ByteArrayRegister}, bytes::lookup_table::builder_operations::ByteLookupOperations, operations::instruction::U32Instructions}, arithmetic::expression, register::{bit::BitRegister, array::ArrayRegister}};
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};

pub struct Keccak256Gadget {
   pub state: ArrayRegister<ByteArrayRegister<8>>,
   pub(crate) round_constant: U64Register,
   pub a: U64Register
}

#[derive(Debug, Clone)]
pub struct Keccak256PublicData<> {
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
    pub fn keccak_f(&mut self,
        operations: &mut ByteLookupOperations,
    ) -> Keccak256Gadget where
    // SHOULD it be u64instructions?
    L::Instruction: U32Instructions {
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
        let round_const = self.alloc::<U64Register>();
        // unsure this is correct..
        // how to constrain it to come constant, i.e. the fixed column for the circuit?
        // for i in 0..24 {
        //     // self.assert_equal(&round_const.get(i)., &ByteArrayRegister::from(KECCAKF_RNDC[i]));
        // }

        // maybe need to boundary the initial state?
        // define boundary constraint for theta
        let state = self.alloc_array::<U64Register>(25);
        let a = self.alloc::<U64Register>();
        for x in 0..5 {
            for y in 0..5 {
                // let res = self.add_u64(&state.get(x + y*5), &a, operations);
                self.set_to_expression_transition(&state.get(x + y*5).next(), state.get(x + y*5).expr() + a.expr());
            }
        }
        self.set_to_expression_transition(&a.next(), a.expr());
        // theta
        // how to constrain an array value follows a transition constraint in for loop? 
        // let c_arr = self.alloc_array::<U64Register>(5);
        // let d_arr = self.alloc_array::<U64Register>(5);
        // for x in 0..5 {
        //     let mut c_i = state.get(x); 
        //     for y in 1..5 {
        //         c_i = self.bitwise_xor(&c_i, &state.get(y * 5 + x), operations);
        //     }
        //     // Does it suffice to constrain for every row of c_i, it need to satisfy its relationship with state at the same row?
        //     self.assert_equal_transition(&c_arr.get(x), &c_i);
        // }
        // // initial state doesn't exist? or don't want to constrain? first row of it is empty?
        // let state_after_theta = self.alloc_array::<U64Register>(25);
        
        // for x in 0..5 {
        //     let temp = self.bit_rotate_right(&c_arr.get((x + 1) % 5), 1, operations);
        //     let d_i = self.bitwise_xor(&c_arr.get((x+4)%5), &temp, operations);
        //     self.assert_equal_transition(&d_arr.get(x), &d_i);
        //     for y in 0..5 {
        //         // make sure state_after_theta follows the theta transition of state
        //         let tmp = self.bitwise_xor(&state.get(y * 5 + x), &d_i, operations);
        //         self.set_to_expression_transition(&(state_after_theta.get(y * 5 + x)), tmp.expr());
        //     }
        // }

        // let state_after_rhopi = self.alloc_array::<U64Register>(25);
        // // 0,0 has no change, direct copy constraint 
        // self.set_to_expression(&state_after_rhopi.get(0), state_after_theta.get(0).expr());
        // // rho and pi
        // for x in 0..5 {
        //     for y in 0..5 {
        //         // x is column, y is row
        //         // y, 2x+3y, is pi_idx in the flatten version
        //         if x+y != 0 {
        //             let pi_idx = ((2 * x + 3 * y) % 5) * 5 + y;
        //             let tmp = self.bit_rotate_right(&state_after_theta.get(y * 5 + x), KECCAKF_ROTC[y][x].try_into().unwrap(), operations);
        //             self.set_to_expression_transition(&state_after_rhopi.get(pi_idx), tmp.expr());
        //         }
        //     }
        // }

        // let state_after_chi = self.alloc_array::<U64Register>(25);
        // // chi
        // for x in 0..5 {
        //     for y in 0..5 {
        //         let tmp1 = self.bitwise_not(&state_after_rhopi.get(  (x + 1) % 5 + y * 5), operations);
        //         let tmp2 = self.bitwise_and(&tmp1, &state_after_rhopi.get((x+2) % 5 + y * 5), operations);
        //         let tmp3 = self.bitwise_xor(&state_after_rhopi.get(x + y * 5), &tmp2, operations);
        //         self.set_to_expression_transition(&state_after_chi.get(x + y*5), tmp3.expr());
        //     }
        // }

        // // iota
        // let tmp = self.bitwise_xor(&state_after_chi.get(0), &round_const, operations);
        // self.set_to_expression_transition(&state.get(0).next(),
        //     tmp.expr());

        // // constrain the other val of next state is the same as chi's result
        // for x in 0..5 {
        //     for y in 0..5 {
        //         // can if used here? if not, guess need to pass some indicator bit to help identity 0,0 
        //         if x + y > 0 {
        //             self.set_to_expression_transition(
        //                 &state.get(x + y * 5).next(), 
        //                      state_after_chi.get(x + y * 5).expr()
        //                 );
        //         }
        //     }
        // }

        Keccak256Gadget {
            a,
            state,
            round_constant: round_const
        }
    }
}

impl Keccak256Gadget {
    pub fn generate_trace<F: Field, I: IntoIterator>(
        &self,
        padded_messages: I,
        writer: &TraceWriter<F>,
    ) -> Keccak256PublicData<>
    where
        I::Item: Borrow<[u8]>
    {
        // TODO: generate the trace so it can satisfy the constraint,
        // note it's tricky since there're many rows where only one column (set of column) is changing, all the other column including state register are basically directly copying to next row, need to carefully compute the gap in the table and accurately put the right value in right row (position), is it correct?....
        // the other cleaner method would be instead of expand to other register(column), use same state register but is the .next() in-place, if not, how to constrain multiple rows, use .next().next() ..? is it clean? 
        // currently there's indeed too many columns, which is bad...for both correct trace generation implementation and performance (proof generation time) i think..
        // also figure out how to pad and discard for variable length input, discard may need some bit array tricks.
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::uint::util::u64_to_le_field_bytes;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Copy)]
    pub struct Keccak256Test;

    impl AirParameters for Keccak256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_FREE_COLUMNS: usize = 1109;
        const EXTENDED_COLUMNS: usize = 2022;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_keccak_256_stark() {
        type F = GoldilocksField;
        type L = Keccak256Test;
        type SC = PoseidonGoldilocksStarkConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut builder = AirBuilder::<L>::new();
       
        let (mut operations, table) = builder.byte_operations();

        let keccak_f_gadget = builder.keccak_f(&mut operations);

        builder.register_byte_lookup(operations, &table);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);
        let writer = generator.new_writer();
        table.write_table_entries(&writer);

        // write the initial value of state to 0th row
        for i in 0..25 {
            writer.write(&keccak_f_gadget.state.get(i), &u64_to_le_field_bytes(0), 0);
        }
        // write the intial a
        writer.write(&keccak_f_gadget.a, &u64_to_le_field_bytes(5), 0);
        // println!("{}", L::num_rows());
        println!("{}", generator.air_data.instructions.len());

        for i in 0..L::num_rows() {
            let round_constant_value = u64_to_le_field_bytes::<F>(KECCAKF_RNDC[i % 24]);
            writer.write(&keccak_f_gadget.round_constant, &round_constant_value, i);
            writer.write_row_instructions(&generator.air_data, i);
        }

        table.write_multiplicities(&writer);

        // after one round result should be at row_index = 1
        

        // for i in 0..L::num_rows() {
        //     println!("{:?}", writer.read(&keccak_f_gadget.round_constant, i));
        // }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        test_starky(&stark, &config, &generator, &[]);

    }
}

