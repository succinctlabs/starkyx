use super::register::SHA256DigestRegister;
use super::SHA256;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::{U32Register, U64Register};
use crate::chip::uint::util::{u32_from_le_field_bytes, u32_to_le_field_bytes};
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::machine::hash::sha::algorithm::SHAir;
use crate::machine::hash::{HashDigest, HashIntConversion, HashInteger};

impl<B: Builder> HashInteger<B> for SHA256 {
    type IntRegister = U32Register;
    type Value = <U32Register as Register>::Value<B::Field>;
}

impl<B: Builder> HashIntConversion<B> for SHA256 {
    fn int_to_field_value(int: Self::Integer) -> Self::Value {
        u32_to_le_field_bytes(int)
    }

    fn field_value_to_int(value: &Self::Value) -> Self::Integer {
        u32_from_le_field_bytes(value)
    }
}

impl<B: Builder> HashDigest<B> for SHA256 {
    type DigestRegister = SHA256DigestRegister;
}

impl<L: AirParameters> SHAir<BytesBuilder<L>, 64> for SHA256
where
    L::Instruction: UintInstructions,
{
    // The state type is the same as the digest type for SHA
    type StateVariable = SHA256DigestRegister;
    type StatePointer = Slice<U64Register>;

    fn clk(builder: &mut BytesBuilder<L>) -> ElementRegister {
        builder.clk
    }

    fn cycles_end_bits(builder: &mut BytesBuilder<L>) -> (BitRegister, BitRegister) {
        let cycle_16 = builder.cycle(4);
        let cycle_64 = builder.cycle(6);

        (cycle_16.end_bit, cycle_64.end_bit)
    }

    fn load_state(
        builder: &mut BytesBuilder<L>,
        hash_state_public: &[Self::StateVariable],
        digest_indices: ArrayRegister<ElementRegister>,
    ) -> Self::StatePointer {
        let state_ptr = builder.uninit_slice();

        for (i, h_slice) in digest_indices.iter().zip(hash_state_public.iter()) {
            for (j, h) in h_slice.split().iter().enumerate() {
                builder.free(&state_ptr.get(j), *h, &Time::from_element(i));
            }
        }

        state_ptr
    }

    fn store_state(
        builder: &mut BytesBuilder<L>,
        state_ptr: &Self::StatePointer,
        state_next: Self::StateVariable,
        time: &Time<L::Field>,
        flag: Option<ElementRegister>,
    ) {
        let state_next = state_next.as_array();
        for i in 0..4 {
            let val = U64Register::from_limbs(&state_next.get_subarray(i * 2..i * 2 + 2));
            builder.store(&state_ptr.get(i), val, time, flag, None, None);
        }
    }

    fn preprocessing_step(
        builder: &mut BytesBuilder<L>,
        w_i_minus_15: Self::IntRegister,
        w_i_minus_2: Self::IntRegister,
        w_i_mimus_16: Self::IntRegister,
        w_i_mimus_7: Self::IntRegister,
    ) -> Self::IntRegister {
        // Calculate the value:
        // s_0 = w_i_minus_15.rotate_right(7) ^ w_i_minus_15.rotate_right(18) ^ (w_i_minus_15 >> 3)
        let w_i_minus_15_rotate_7 = builder.rotate_right(w_i_minus_15, 7);
        let w_i_minus_15_rotate_18 = builder.rotate_right(w_i_minus_15, 18);
        let w_i_minus_15_shr_3 = builder.shr(w_i_minus_15, 3);

        let mut s_0 = builder.xor(&w_i_minus_15_rotate_7, &w_i_minus_15_rotate_18);
        s_0 = builder.xor(&s_0, &w_i_minus_15_shr_3);

        // Calculate the value:
        // s_1 = w_i_minus_2.rotate_right(17) ^ w_i_minus_2.rotate_right(19) ^ (w_i_minus_2 >> 10)
        let w_i_minus_2_rotate_17 = builder.rotate_right(w_i_minus_2, 17);
        let w_i_minus_2_rotate_19 = builder.rotate_right(w_i_minus_2, 19);
        let w_i_minus_2_shr_10 = builder.shr(w_i_minus_2, 10);

        let mut s_1 = builder.xor(&w_i_minus_2_rotate_17, &w_i_minus_2_rotate_19);
        s_1 = builder.xor(&s_1, &w_i_minus_2_shr_10);

        // Calculate the value:
        // w_i = w_i_minus_16 + s_0 + w_i_minus_7 + s_1
        let mut w_i_pre_process = builder.add(w_i_mimus_16, s_0);
        w_i_pre_process = builder.add(w_i_pre_process, w_i_mimus_7);
        builder.add(w_i_pre_process, s_1)
    }

    fn processing_step(
        builder: &mut BytesBuilder<L>,
        vars: ArrayRegister<Self::IntRegister>,
        w_i: Self::IntRegister,
        round_constant: Self::IntRegister,
    ) -> Vec<Self::IntRegister> {
        let a = vars.get(0);
        let b = vars.get(1);
        let c = vars.get(2);
        let d = vars.get(3);
        let e = vars.get(4);
        let f = vars.get(5);
        let g = vars.get(6);
        let h = vars.get(7);

        // Calculate sum_1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25).
        let e_rotate_6 = builder.rotate_right(e, 6);
        let e_rotate_11 = builder.rotate_right(e, 11);
        let e_rotate_25 = builder.rotate_right(e, 25);
        let mut sum_1 = builder.xor(e_rotate_6, e_rotate_11);
        sum_1 = builder.xor(sum_1, e_rotate_25);

        // Calculate ch = (e & f) ^ (!e & g).
        let e_and_f = builder.and(&e, &f);
        let not_e = builder.not(e);
        let not_e_and_g = builder.and(&not_e, &g);
        let ch = builder.xor(&e_and_f, &not_e_and_g);

        // Calculate temp_1 = h + sum_1 + ch + round_constant + w.
        let mut temp_1 = builder.add(h, sum_1);
        temp_1 = builder.add(temp_1, ch);
        temp_1 = builder.add(temp_1, round_constant);
        temp_1 = builder.add(temp_1, w_i);

        // Calculate sum_0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22).
        let a_rotate_2 = builder.rotate_right(a, 2);
        let a_rotate_13 = builder.rotate_right(a, 13);
        let a_rotate_22 = builder.rotate_right(a, 22);
        let mut sum_0 = builder.xor(a_rotate_2, a_rotate_13);
        sum_0 = builder.xor(sum_0, a_rotate_22);

        // Calculate maj = (a & b) ^ (a & c) ^ (b & c);
        let a_and_b = builder.and(a, b);
        let a_and_c = builder.and(a, c);
        let b_and_c = builder.and(b, c);
        let mut maj = builder.xor(a_and_b, a_and_c);
        maj = builder.xor(maj, b_and_c);

        // Calculate temp_2 = sum_0 + maj.
        let temp_2 = builder.add(sum_0, maj);

        // Calculate the next cycle values.
        let a_next = builder.add(temp_1, temp_2);
        let b_next = a;
        let c_next = b;
        let d_next = c;
        let e_next = builder.add(d, temp_1);
        let f_next = e;
        let g_next = f;
        let h_next = g;

        vec![
            a_next, b_next, c_next, d_next, e_next, f_next, g_next, h_next,
        ]
    }

    fn absorb(
        builder: &mut BytesBuilder<L>,
        state: ArrayRegister<Self::IntRegister>,
        vars_next: &[Self::IntRegister],
    ) -> Self::StateVariable {
        let state_next = builder.alloc_array(8);
        for ((s, v), res) in state.iter().zip(vars_next.iter()).zip(state_next.iter()) {
            let carry = builder.alloc();
            builder
                .api
                .set_add_u32(&s, v, &None, &res, &carry, &mut builder.operations)
        }
        Self::StateVariable::from_array(state_next)
    }
}

#[cfg(test)]
mod tests {
    use core::iter;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::machine::hash::sha::builder::test_utils::test_sha;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SHA256Test;

    impl AirParameters for SHA256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 418;
        const EXTENDED_COLUMNS: usize = 912;
    }

    fn test_sha256<'a, I: IntoIterator<Item = &'a [u8]>, J: IntoIterator<Item = &'a str>>(
        messages: I,
        expected_digests: J,
    ) {
        test_sha::<SHA256Test, SHA256, _, _, 64>(messages, expected_digests)
    }

    #[test]
    fn test_sha256_short_message() {
        let msg = b"abc";
        let expected_digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let num_messages = 2;
        test_sha256(
            iter::repeat(msg).take(num_messages).map(|x| x.as_slice()),
            iter::repeat(expected_digest).take(num_messages),
        )
    }

    #[test]
    fn test_sha256_long_message() {
        let num_messages = 1023;
        let msg = hex::decode("243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917").unwrap();
        let expected_digest = "aca16131a2e4c4c49e656d35aac1f0e689b3151bb108fa6cf5bcc3ac08a09bf9";
        test_sha256(
            iter::repeat(msg.as_slice()).take(num_messages),
            iter::repeat(expected_digest).take(num_messages),
        )
    }

    #[test]
    fn test_sha256_changing_length_nessage() {
        let short_msg = b"abc";
        let short_expected_digest =
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let long_msg = hex::decode("243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b5470917").unwrap();
        let long_expected_digest =
            "aca16131a2e4c4c49e656d35aac1f0e689b3151bb108fa6cf5bcc3ac08a09bf9";
        test_sha256(
            [
                short_msg.as_slice(),
                long_msg.as_slice(),
                short_msg.as_slice(),
            ],
            [
                short_expected_digest,
                long_expected_digest,
                short_expected_digest,
            ],
        );
    }
}
