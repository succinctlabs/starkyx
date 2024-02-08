use super::register::SHA512DigestRegister;
use super::SHA512;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::{u64_from_le_field_bytes, u64_to_le_field_bytes};
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::machine::hash::sha::algorithm::SHAir;
use crate::machine::hash::{HashDigest, HashIntConversion, HashInteger};

impl<B: Builder> HashInteger<B> for SHA512 {
    type Value = <U64Register as Register>::Value<B::Field>;
    type IntRegister = U64Register;
}

impl<B: Builder> HashIntConversion<B> for SHA512 {
    fn int_to_field_value(int: Self::Integer) -> Self::Value {
        u64_to_le_field_bytes(int)
    }

    fn field_value_to_int(value: &Self::Value) -> Self::Integer {
        u64_from_le_field_bytes(value)
    }
}

impl<B: Builder> HashDigest<B> for SHA512 {
    type DigestRegister = SHA512DigestRegister;
}

impl<L: AirParameters> SHAir<BytesBuilder<L>, 80> for SHA512
where
    L::Instruction: UintInstructions,
{
    // The state type is the same as the digest type for SHA
    type StateVariable = SHA512DigestRegister;
    type StatePointer = Slice<U64Register>;

    fn clk(builder: &mut BytesBuilder<L>) -> ElementRegister {
        builder.clk
    }

    fn cycles_end_bits(builder: &mut BytesBuilder<L>) -> (BitRegister, BitRegister) {
        let cycle_16 = builder.cycle(4);
        let cycle_80_end_bit = {
            let loop_5 = builder.api.loop_instr(5);
            let five_end_bit = loop_5.get_iteration_reg(4);
            builder.mul(five_end_bit, cycle_16.end_bit)
        };

        (cycle_16.end_bit, cycle_80_end_bit)
    }

    fn load_state(
        builder: &mut BytesBuilder<L>,
        hash_state_public: &[Self::StateVariable],
        digest_indices: ArrayRegister<ElementRegister>,
    ) -> Self::StatePointer {
        let state_ptr = builder.uninit_slice();

        for (i, h_slice) in digest_indices.iter().zip(hash_state_public.iter()) {
            for (j, h) in h_slice.iter().enumerate() {
                builder.free(&state_ptr.get(j), h, &Time::from_element(i));
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
        for (i, element) in state_next.iter().enumerate() {
            builder.store(&state_ptr.get(i), element, time, flag, None, None);
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
        // s_0 = w_i_minus_15.rotate_right(1) ^ w_i_minus_15.rotate_right(8) ^ (w_i_minus_15 >> 7)
        let w_i_minus_15_rotate_1 = builder.rotate_right(w_i_minus_15, 1);
        let w_i_minus_15_rotate_8 = builder.rotate_right(w_i_minus_15, 8);
        let w_i_minus_15_shr_7 = builder.shr(w_i_minus_15, 7);

        let mut s_0 = builder.xor(&w_i_minus_15_rotate_1, &w_i_minus_15_rotate_8);
        s_0 = builder.xor(&s_0, &w_i_minus_15_shr_7);

        let w_i_minus_2_rotate_19 = builder.rotate_right(w_i_minus_2, 19);
        let w_i_minus_2_rotate_61 = builder.rotate_right(w_i_minus_2, 61);
        let w_i_minus_2_shr_6 = builder.shr(w_i_minus_2, 6);

        let mut s_1 = builder.xor(&w_i_minus_2_rotate_19, &w_i_minus_2_rotate_61);
        s_1 = builder.xor(&s_1, &w_i_minus_2_shr_6);

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

        // Calculate S1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41).
        let e_rotate_14 = builder.rotate_right(e, 14);
        let e_rotate_18 = builder.rotate_right(e, 18);
        let e_rotate_41 = builder.rotate_right(e, 41);
        let mut sum_1 = builder.xor(e_rotate_14, e_rotate_18);
        sum_1 = builder.xor(sum_1, e_rotate_41);

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

        // Calculate S0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39).
        let a_rotate_28 = builder.rotate_right(a, 28);
        let a_rotate_34 = builder.rotate_right(a, 34);
        let a_rotate_39 = builder.rotate_right(a, 39);
        let mut sum_0 = builder.xor(a_rotate_28, a_rotate_34);
        sum_0 = builder.xor(sum_0, a_rotate_39);

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
        for ((s, v), sum) in state.iter().zip(vars_next.iter()).zip(state_next.iter()) {
            let carry = builder.alloc();
            builder
                .api
                .set_add_u64(&s, v, &None, &sum, &carry, &mut builder.operations)
        }
        Self::StateVariable::from_array(state_next)
    }
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;
    use core::iter;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use serde::{Deserialize, Serialize};

    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::AirParameters;
    use crate::machine::hash::sha::builder::test_utils::test_sha;
    use crate::machine::hash::sha::sha512::SHA512;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct SHA512Test;

    impl AirParameters for SHA512Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 815;
        const EXTENDED_COLUMNS: usize = 1782;
    }

    fn test_sha512<'a, I: IntoIterator<Item = &'a [u8]>, J: IntoIterator<Item = &'a str>>(
        messages: I,
        expected_digests: J,
    ) {
        test_sha::<SHA512Test, SHA512, _, _, 80>(messages, expected_digests)
    }

    #[test]
    fn test_sha512_short_message() {
        let msg = b"plonky2";
        let expected_digest = "7c6159dd615db8c15bc76e23d36106e77464759979a0fcd1366e531f552cfa0852dbf5c832f00bb279cbc945b44a132bff3ed0028259813b6a07b57326e88c87";
        let num_messages = 2;
        test_sha512(
            iter::repeat(msg).take(num_messages).map(|x| x.as_slice()),
            iter::repeat(expected_digest).take(num_messages),
        )
    }

    #[test]
    fn test_sha512_long_message() {
        let num_messages = 600;
        let msg = hex::decode("35c323757c20640a294345c89c0bfcebe3d554fdb0c7b7a0bdb72222c531b1ecf7ec1c43f4de9d49556de87b86b26a98942cb078486fdb44de38b80864c3973153756363696e6374204c616273").unwrap();
        let expected_digest = "4388243c4452274402673de881b2f942ff5730fd2c7d8ddb94c3e3d789fb3754380cba8faa40554d9506a0730a681e88ab348a04bc5c41d18926f140b59aed39";
        test_sha512(
            iter::repeat(msg.as_slice()).take(num_messages),
            iter::repeat(expected_digest).take(num_messages),
        )
    }

    #[test]
    fn test_sha512_changing_length_nessage() {
        let short_msg = b"plonky2";
        let short_expected_digest = "7c6159dd615db8c15bc76e23d36106e77464759979a0fcd1366e531f552cfa0852dbf5c832f00bb279cbc945b44a132bff3ed0028259813b6a07b57326e88c87";
        let long_msg = hex::decode("35c323757c20640a294345c89c0bfcebe3d554fdb0c7b7a0bdb72222c531b1ecf7ec1c43f4de9d49556de87b86b26a98942cb078486fdb44de38b80864c3973153756363696e6374204c616273").unwrap();
        let long_expected_digest = "4388243c4452274402673de881b2f942ff5730fd2c7d8ddb94c3e3d789fb3754380cba8faa40554d9506a0730a681e88ab348a04bc5c41d18926f140b59aed39";
        test_sha512(
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
