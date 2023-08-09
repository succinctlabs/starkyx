use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::uint::operations::instruction::U32Instructions;
use crate::chip::uint::register::U32Register;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub fn round_constants<F: Field>() -> [[F; 4]; 64] {
    [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ]
    .map(u32::to_le_bytes)
    .map(|x| x.map(F::from_canonical_u8))
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl<L: AirParameters> AirBuilder<L> {
    pub fn sha_256_round(&mut self, chunk: ArrayRegister<U32Register>)
    where
        L::Instruction: U32Instructions,
    {
        let cycle_64 = self.cycle(6);
        let cycle_16 = self.cycle(4);

        // Inistialize the byte lookup table
        let (mut operations, mut table) = self.byte_operations();

        // Absorbe values into the sponge
        let w_value = self.alloc::<U32Register>();
        let ab_s_0 = self.alloc::<U32Register>();
        let ab_s_1 = self.alloc::<U32Register>();

        // Put the w value into the bus
        //TODO

        // Sponge compression
        let a = self.alloc::<U32Register>();
        let b = self.alloc::<U32Register>();
        let c = self.alloc::<U32Register>();
        let d = self.alloc::<U32Register>();
        let e = self.alloc::<U32Register>();
        let f = self.alloc::<U32Register>();
        let g = self.alloc::<U32Register>();
        let h = self.alloc::<U32Register>();

        // Get the round constants
        let round_constants = round_constants::<L::Field>();

        // Calculate sum_1 = (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25)
        let e_rot_6 = self.bit_rotate_right(&e, 6, &mut operations);
        let e_rot_11 = self.bit_rotate_right(&e, 11, &mut operations);
        let e_rot_25 = self.bit_rotate_right(&e, 25, &mut operations);
        let mut sum_1 = self.bitwise_xor(&e_rot_6, &e_rot_11, &mut operations);
        sum_1 = self.bitwise_xor(&sum_1, &e_rot_25, &mut operations);

        // Calculate ch := (e and f) xor ((not e) and g)
        let e_and_f = self.bitwise_and(&e, &f, &mut operations);
        let not_e = self.bitwise_not(&e, &mut operations);
        let not_e_and_g = self.bitwise_and(&not_e, &g, &mut operations);
        let ch = self.bitwise_xor(&e_and_f, &not_e_and_g, &mut operations);
    }
}
