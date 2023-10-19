use super::data::SHA256Data;
use crate::chip::memory::time::Time;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U32Register;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;

impl<L: AirParameters> BytesBuilder<L>
where
    L::Instruction: UintInstructions,
{
    pub fn sha_256_preprocessing(
        &mut self,
        data: &SHA256Data,
        is_active: BitRegister,
    ) -> U32Register {
        let w = &data.w;
        let time = Time::from_element(data.process_id);

        // Calculate s_0 = w_i_minus_15.rotate_right(7) ^ w_i_minus_15.rotate_right(18) ^ (w_i_minus_15 >> 3);
        let i_m_15 = self.alloc::<ElementRegister>();
        let w_i_minus_15 = self.get(&w.get_at(i_m_15), &time);
        let w_i_minus_15_rotate_7 = self.rotate_right(w_i_minus_15, 7);
        let w_i_minus_15_rotate_18 = self.rotate_right(w_i_minus_15, 18);
        let w_i_minus_15_shr_3 = self.shr(w_i_minus_15, 3);

        let mut s_0 = self.xor(&w_i_minus_15_rotate_7, &w_i_minus_15_rotate_18);
        s_0 = self.xor(&s_0, &w_i_minus_15_shr_3);

        // Calculate s_1 = w_i_minus_2.rotate_right(17) ^ w_i_minus_2.rotate_right(19) ^ (w_i_minus_2 >> 10);
        let i_m_2 = self.alloc::<ElementRegister>();
        let w_i_minus_2 = self.get(&w.get_at(i_m_2), &time);
        let w_i_minus_2_rotate_17 = self.rotate_right(w_i_minus_2, 17);
        let w_i_minus_2_rotate_19 = self.rotate_right(w_i_minus_2, 19);
        let w_i_minus_2_shr_10 = self.shr(w_i_minus_2, 10);

        let mut s_1 = self.xor(&w_i_minus_2_rotate_17, &w_i_minus_2_rotate_19);
        s_1 = self.xor(&s_1, &w_i_minus_2_shr_10);

        // Calculate w_i = w_i_minus_16 + s_0 + w_i_minus_7 + s_1;
        let i_m_16 = self.alloc::<ElementRegister>();
        let i_m_7 = self.alloc::<ElementRegister>();
        let w_i_mimus_16 = self.get(&w.get_at(i_m_16), &time);
        let w_i_mimus_7 = self.get(&w.get_at(i_m_7), &time);
        let mut w_i = self.add(w_i_mimus_16, s_0);
        w_i = self.add(w_i, w_i_mimus_7);
        w_i = self.add(w_i, s_1);
        // self.assert_expression_zero(w_bit.not_expr() * (w_i.expr() - w_window.get(0).expr()));

        todo!()
    }
}
