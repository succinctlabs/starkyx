use core::borrow::Borrow;

use plonky2::field::types::Field;

use crate::chip::{builder::{AirBuilder}, AirParameters, trace::writer::TraceWriter, uint::{register::{U64Register, ByteArrayRegister}, bytes::lookup_table::builder_operations::ByteLookupOperations, operations::instruction::U32Instructions}, arithmetic::expression, register::bit::BitRegister};
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};

use super::sha256::U32Value;

pub struct Keccak256Gadget {
   
}

#[derive(Debug, Clone)]
pub struct Keccak256PublicData<T> {
    pub public_w: Vec<U32Value<T>>,
    pub hash_state: Vec<U32Value<T>>,
    pub end_bits: Vec<T>,
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
pub const KECCAKF_RNDC: [[u32; 2]; 24] = [
    [0x00000001, 0x00000000], [0x00008082, 0x00000000],
    [0x0000808A, 0x80000000], [0x80008000, 0x80000000],
    [0x0000808B, 0x00000000], [0x80000001, 0x00000000],
    [0x80008081, 0x80000000], [0x00008009, 0x80000000],
    [0x0000008A, 0x00000000], [0x00000088, 0x00000000],
    [0x80008009, 0x00000000], [0x8000000A, 0x00000000],
    [0x8000808B, 0x00000000], [0x0000008B, 0x80000000],
    [0x00008089, 0x80000000], [0x00008003, 0x80000000],
    [0x00008002, 0x80000000], [0x00000080, 0x80000000],
    [0x0000800A, 0x00000000], [0x8000000A, 0x80000000],
    [0x80008081, 0x80000000], [0x00008080, 0x80000000],
    [0x80000001, 0x00000000], [0x80008008, 0x80000000],
];

// impl constraint for keccack256 
// state register 
impl<L: AirParameters> AirBuilder<L> {
    pub fn keccak_f(&mut self,
        operations: &mut ByteLookupOperations,
    )where
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

        let round_const = self.alloc_array::<U64Register>(24);
        // unsure this is correct..
        // how to constrain it to come constant, i.e. the fixed column for the circuit?
        for i in 0..24 {
            // self.assert_equal(&round_const.get(i)., &ByteArrayRegister::from(KECCAKF_RNDC[i]));
        }

        // maybe need to boundary the initial state?
        // define boundary constraint for theta
        let state = self.alloc_array::<U64Register>(25);
        
        // theta
        // how to constrain an array value follows a transition constraint in for loop? 
        let c_arr = self.alloc_array::<U64Register>(5);
        let d_arr = self.alloc_array::<U64Register>(5);
        for x in 0..5 {
            let mut c_i = state.get(x); 
            for y in 1..5 {
                c_i = self.bitwise_xor(&c_i, &state.get(y * 5 + x), operations);
            }
            // Does it suffice to constrain for every row of c_i, it need to satisfy its relationship with state at the same row?
            self.assert_equal_transition(&c_arr.get(x), &c_i);
        }
        // initial state doesn't exist? or don't want to constrain? first row of it is empty?
        let state_after_theta = self.alloc_array::<U64Register>(25);
        
        for x in 0..5 {
            let temp = self.bit_rotate_right(&c_arr.get((x + 1) % 5), 1, operations);
            let d_i = self.bitwise_xor(&c_arr.get((x+4)%5), &temp, operations);
            self.assert_equal_transition(&d_arr.get(x), &d_i);
            for y in 0..5 {
                // make sure next row of state_after_theta follows the theta transition of state
                let tmp = self.bitwise_xor(&state.get(y * 5 + x), &d_i, operations);
                self.set_to_expression_transition(&(state_after_theta.get(y * 5 + x)).next(), tmp.expr());
            }
        }

        let state_after_rhopi = self.alloc_array::<U64Register>(25);

        // rho and pi
        for x in 0..5 {
            for y in 0..5 {
                // x is column, y is row
                // y, 2x+3y, is pi_idx in the flatten version
                let pi_idx = ((2 * x + 3 * y) % 5) * 5 + y;
                let tmp = self.bit_rotate_right(&state_after_theta.get(y * 5 + x), KECCAKF_ROTC[y][x].try_into().unwrap(), operations);
                self.set_to_expression_transition(&state_after_rhopi.get(pi_idx).next(), tmp.expr());
            }
        }

        let state_after_chi = self.alloc_array::<U64Register>(25);
        // chi
        for x in 0..5 {
            for y in 0..5 {
              
                let tmp1 = self.bitwise_not(&state_after_rhopi.get(  (x + 1) % 5 + y * 5), operations);
                let tmp2 = self.bitwise_and(&tmp1, &state_after_rhopi.get((x+2) % 5 + y * 5), operations);
                let tmp3 = self.bitwise_xor(&state_after_rhopi.get(x + y * 5), &tmp2, operations);
                self.set_to_expression_transition(&state_after_chi.get(x + y*5), tmp3.expr());
            }
        }

        // iota
        for i in 0..24 {
            let tmp = self.bitwise_xor(&state_after_chi.get(0), &round_const.get(i), operations);
            self.set_to_expression_transition(&state.get(0).next(),
            tmp.expr());
        }

        // constrain the other val of next state is the same as chi's result
        for x in 0..5 {
            for y in 0..5 {
                // can if used here? if not, guess need to pass some indicator bit to help identity 0,0 
                if x + y > 0 {
                    self.set_to_expression_transition(
                        &state.get(x + y * 5).next(), 
                             state_after_chi.get(x + y * 5).expr()
                        );
                }
               
            }
        }
    }
}

impl Keccak256Gadget {
    pub fn generate_trace<F: Field, I: IntoIterator>(
        &self,
        padded_messages: I,
        writer: &TraceWriter<F>,
    ) -> Keccak256PublicData<F>
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
