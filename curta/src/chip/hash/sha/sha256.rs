use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::bytes::lookup_table::table::ByteLookupTable;
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

    pub fn round_constants(&mut self) -> U32Register {
        let round_constant = self.alloc::<U32Register>();

        round_constant
    } 

    pub fn sha_256_process(
        &mut self,
        initial_hash: &ArrayRegister<U32Register>,
        round_constant : &U32Register,
        w: &U32Register,
        w_filter: &BitRegister,
        w_challenges: &ArrayRegister<CubicRegister>,
        idx: usize,
    ) -> ByteLookupTable<L::Field>
    where
        L::Instruction: U32Instructions,
    {
        let cycle_64 = self.cycle(6);
        let cycle_16 = self.cycle(4);
        let clk = self.clock();

        // Inistialize the byte lookup table
        let (mut operations, mut table) = self.byte_operations();

        // Absorbe values into the sponge
        let w_i_minus_2 = self.alloc::<U32Register>();
        let w_i_minus_7 = self.alloc::<U32Register>();
        let w_i_minus_15 = self.alloc::<U32Register>();
        let w_i_minus_16 = self.alloc::<U32Register>();

        // Get the values from the bus
        let two = ArithmeticExpression::from_constant(L::Field::from_canonical_u8(2));
        let clk_w_i_minus_2 =
            self.accumulate_expressions(&w_challenges, &[clk.expr() - two, w_i_minus_2.expr()]);
        self.output_from_bus_filtered(idx, clk_w_i_minus_2, w_filter.expr());

        let seven = ArithmeticExpression::from_constant(L::Field::from_canonical_u8(7));
        let clk_w_i_minus_7 =
            self.accumulate_expressions(&w_challenges, &[clk.expr() - seven, w_i_minus_7.expr()]);
        self.output_from_bus_filtered(idx, clk_w_i_minus_7, w_filter.expr());

        let fifteen = ArithmeticExpression::from_constant(L::Field::from_canonical_u8(15));
        let clk_w_i_minus_15 = self
            .accumulate_expressions(&w_challenges, &[clk.expr() - fifteen, w_i_minus_15.expr()]);
        self.output_from_bus_filtered(idx, clk_w_i_minus_15, w_filter.expr());

        let sixteen = ArithmeticExpression::from_constant(L::Field::from_canonical_u8(16));
        let clk_w_i_minus_16 = self
            .accumulate_expressions(&w_challenges, &[clk.expr() - sixteen, w_i_minus_16.expr()]);
        self.output_from_bus_filtered(idx, clk_w_i_minus_16, w_filter.expr());

        // Compute s_0 = (w_i-15 rot 7) xor (w_i-15 rot 18) xor (w_i-15 shr 3)
        let w_i_minus_15_rot_7 = self.bit_rotate_right(&w_i_minus_15, 7, &mut operations);
        let w_i_minus_15_rot_18 = self.bit_rotate_right(&w_i_minus_15, 18, &mut operations);
        let w_i_minus_15_shr_3 = self.bit_shr(&w_i_minus_15, 3, &mut operations);
        let mut s_0 = self.bitwise_xor(&w_i_minus_15_rot_7, &w_i_minus_15_rot_18, &mut operations);
        s_0 = self.bitwise_xor(&s_0, &w_i_minus_15_shr_3, &mut operations);

        // Compute s_1 = (w_i-2 rot 17) xor (w_i-2 rot 19) xor (w_i-2 shr 10)
        let w_i_minus_2_rot_17 = self.bit_rotate_right(&w_i_minus_2, 17, &mut operations);
        let w_i_minus_2_rot_19 = self.bit_rotate_right(&w_i_minus_2, 19, &mut operations);
        let w_i_minus_2_shr_10 = self.bit_shr(&w_i_minus_2, 10, &mut operations);
        let mut s_1 = self.bitwise_xor(&w_i_minus_2_rot_17, &w_i_minus_2_rot_19, &mut operations);
        s_1 = self.bitwise_xor(&s_1, &w_i_minus_2_shr_10, &mut operations);

        // Compute w_i = w_i-16 + s_0 + w_i-7 + s_1
        let mut w_i = self.add_u32(&w_i_minus_16, &s_0, &mut operations);
        w_i = self.add_u32(&w_i, &w_i_minus_7, &mut operations);
        w_i = self.add_u32(&w_i, &s_1, &mut operations);

        // Assign the w_i values to the w register
        // We want to impose w_i = w_register for i = 16..64 in every cycle.
        let w_filter = self.alloc::<BitRegister>();
        // assert that w_filter = 0 in the begining on each cycle
        self.assert_expression_zero(w_filter.expr() * cycle_64.start_bit.expr());
        // assert that w_filter = 1 if w_filter = 0 and same otherwise at every 16 cycle)
        self.assert_expression_zero_transition(
            (w_filter.next().expr() - ArithmeticExpression::one())
                * cycle_64.end_bit.expr()
                * (ArithmeticExpression::one() - cycle_64.end_bit.expr()),
        );
        self.assert_expression_zero((w_i.expr() - w.expr()) * w_filter.expr());

        // Output w from the bus to compare with the input
        let clk_w = self.accumulate_expressions(&w_challenges, &[clk.expr(), w.expr()]);
        self.output_from_bus_filtered(idx, clk_w, w_filter.not_expr());        // Hash state
        let state = self.alloc_array::<U32Register>(8);
        // set the hash state to the initial hash on first row
        for (h_i, initial_hash_i) in state.iter().zip(initial_hash.iter()) {
            self.set_to_expression_first_row(&h_i, initial_hash_i.expr());
        }        
        
        // Sponge compression
        let a = self.alloc::<U32Register>();
        let b = self.alloc::<U32Register>();
        let c = self.alloc::<U32Register>();
        let d = self.alloc::<U32Register>();
        let e = self.alloc::<U32Register>();
        let f = self.alloc::<U32Register>();
        let g = self.alloc::<U32Register>();
        let h = self.alloc::<U32Register>();

        // Set (a, b, .. , h) to be the initial hash values on the first row
        self.set_to_expression_first_row(&a, initial_hash.get(0).expr());
        self.set_to_expression_first_row(&b, initial_hash.get(1).expr());
        self.set_to_expression_first_row(&c, initial_hash.get(2).expr());
        self.set_to_expression_first_row(&d, initial_hash.get(3).expr());
        self.set_to_expression_first_row(&e, initial_hash.get(4).expr());
        self.set_to_expression_first_row(&f, initial_hash.get(5).expr());
        self.set_to_expression_first_row(&g, initial_hash.get(6).expr());
        self.set_to_expression_first_row(&h, initial_hash.get(7).expr());



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

        // Calculate temp_1 = h + sum_1 + ch + round_constant + w
        let mut temp_1 = self.add_u32(&sum_1, &h, &mut operations);
        temp_1 = self.add_u32(&temp_1, &ch, &mut operations);
        temp_1 = self.add_u32(&temp_1, &round_constant, &mut operations);
        temp_1 = self.add_u32(&temp_1, &w, &mut operations);

        // Calculate sum_0 = (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22)
        let a_rot_2 = self.bit_rotate_right(&a, 2, &mut operations);
        let a_rot_13 = self.bit_rotate_right(&a, 13, &mut operations);
        let a_rot_22 = self.bit_rotate_right(&a, 22, &mut operations);
        let mut sum_0 = self.bitwise_xor(&a_rot_2, &a_rot_13, &mut operations);
        sum_0 = self.bitwise_xor(&sum_0, &a_rot_22, &mut operations);

        // Calculate maj := (a and b) xor (a and c) xor (b and c)
        let a_and_b = self.bitwise_and(&a, &b, &mut operations);
        let a_and_c = self.bitwise_and(&a, &c, &mut operations);
        let b_and_c = self.bitwise_and(&b, &c, &mut operations);
        let maj = self.bitwise_xor(&a_and_b, &a_and_c, &mut operations);
        let maj = self.bitwise_xor(&maj, &b_and_c, &mut operations);

        // Calculate temp_2 = sum_0 + maj
        let temp_2 = self.add_u32(&sum_0, &maj, &mut operations);

        // Set the next row values
        let a_next = self.add_u32(&temp_1, &temp_2, &mut operations);
        let b_next = a;
        let c_next = b;
        let d_next = c;
        let e_next = self.add_u32(&d, &temp_1, &mut operations);
        let f_next = e;
        let g_next = f;
        let h_next = g;

        // Add the working variables to the state at the end of the round
        let a_plus_h_0 = self.add_u32(&a_next, &state.get(0), &mut operations).expr();
        let b_plus_h_1 = self.add_u32(&b_next, &state.get(1), &mut operations).expr();
        let c_plus_h_2 = self.add_u32(&c_next, &state.get(2), &mut operations).expr();
        let d_plus_h_3 = self.add_u32(&d_next, &state.get(3), &mut operations).expr();
        let e_plus_h_4 = self.add_u32(&e_next, &state.get(4), &mut operations).expr();
        let f_plus_h_5 = self.add_u32(&f_next, &state.get(5), &mut operations).expr();
        let g_plus_h_6 = self.add_u32(&g_next, &state.get(6), &mut operations).expr();
        let h_plus_h_7 = self.add_u32(&h_next, &state.get(7), &mut operations).expr();

        let end_bit = cycle_64.end_bit;
        self.set_to_expression_transition(
            &state.get(0).next(),
            end_bit.not_expr() * state.get(0).expr() + end_bit.expr() * a_plus_h_0,
        );
        self.set_to_expression_transition(
            &state.get(1).next(),
            end_bit.not_expr() * state.get(1).expr() + end_bit.expr() * b_plus_h_1,
        );
        self.set_to_expression_transition(
            &state.get(2).next(),
            end_bit.not_expr() * state.get(2).expr() + end_bit.expr() * c_plus_h_2,
        );
        self.set_to_expression_transition(
            &state.get(3).next(),
            end_bit.not_expr() * state.get(3).expr() + end_bit.expr() * d_plus_h_3,
        );
        self.set_to_expression_transition(
            &state.get(4).next(),
            end_bit.not_expr() * state.get(4).expr() + end_bit.expr() * e_plus_h_4,
        );
        self.set_to_expression_transition(
            &state.get(5).next(),
            end_bit.not_expr() * state.get(5).expr() + end_bit.expr() * f_plus_h_5,
        );
        self.set_to_expression_transition(
            &state.get(6).next(),
            end_bit.not_expr() * state.get(6).expr() + end_bit.expr() * g_plus_h_6,
        );
        self.set_to_expression_transition(
            &state.get(7).next(),
            end_bit.not_expr() * state.get(7).expr() + end_bit.expr() * h_plus_h_7,
        );

        self.set_to_expression_transition(
            &a.next(),
            end_bit.expr() * state.get(0).expr() + end_bit.not_expr() * a_next.expr(),
        );
        self.set_to_expression_transition(
            &b.next(),
            end_bit.expr() * state.get(1).expr() + end_bit.not_expr() * b_next.expr(),
        );
        self.set_to_expression_transition(
            &c.next(),
            end_bit.expr() * state.get(2).expr() + end_bit.not_expr() * c_next.expr(),
        );
        self.set_to_expression_transition(
            &d.next(),
            end_bit.expr() * state.get(3).expr() + end_bit.not_expr() * d_next.expr(),
        );
        self.set_to_expression_transition(
            &e.next(),
            end_bit.expr() * state.get(4).expr() + end_bit.not_expr() * e_next.expr(),
        );
        self.set_to_expression_transition(
            &f.next(),
            end_bit.expr() * state.get(5).expr() + end_bit.not_expr() * f_next.expr(),
        );
        self.set_to_expression_transition(
            &g.next(),
            end_bit.expr() * state.get(6).expr() + end_bit.not_expr() * g_next.expr(),
        );
        self.set_to_expression_transition(
            &h.next(),
            end_bit.expr() * state.get(7).expr() + end_bit.not_expr() * h_next.expr(),
        );

        // Dummpy operation
        let dummy = self.bit_shr(&a, 9, &mut operations);
        self.register_byte_lookup(operations, &mut table);

        table
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

    #[test]
    fn test_sha_256_stark() {
        type F = GoldilocksField;
        type L = SHA256Test;
        type SC = PoseidonGoldilocksStarkConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Sha256 test", log::Level::Debug);

        let mut builder = AirBuilder::<L>::new();
        let mut bus = builder.new_bus();
        let idx = bus.new_channel(&mut builder);

        let initial_hash = builder.alloc_array_public::<U32Register>(8);
        let round_constant = builder.alloc::<U32Register>();
        let w = builder.alloc::<U32Register>();
        let w_filter = builder.alloc::<BitRegister>();
        let w_challenges =
            builder.alloc_challenge_array::<CubicRegister>(1 + U32Register::size_of());        
            
        // Alloc round constant
        // let round_constant_challenges = builder.alloc_challenge_array(U32Register::size_of() + 1);
        // let round_constant_digest = builder.accumulate_expressions(
        //     &round_constant_challenges,
        //     &[clk.expr(), round_constant.expr()],
        // );
        // builder.output_from_bus(idx, round_constant_digest);

        let mut table = builder.sha_256_process(
            &initial_hash,
            &round_constant,
            &w,
            &w_filter,
            &w_challenges,
            idx,
        );

        // builder.constrain_bus(bus);
        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        table.write_table_entries(&writer);

        // Write public inputs
        let initial_hash_value = first_hash_value::<F>();
        writer.write_array(&initial_hash, &initial_hash_value, 0);
        let round_constants_value = round_constants::<F>();
        // writer.write_array(&round_constants_reg, &round_constants_value, 0);

        let public_inputs = writer.public.read().unwrap().clone();

        rayon::join(
            || {
                for i in 0..L::num_rows() {
                    if i % 64 < 16 {
                        writer.write(&w_filter, &F::ZERO, i);
                    } else {
                        writer.write(&w_filter, &F::ONE, i);
                    }
                    writer.write(&round_constant, &round_constants_value[i % 64], i);
                    writer.write_row_instructions(&air, i);
                }
            },
            || table.write_multiplicities(&writer),
        );

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        timed!(
            timing,
            "test starky",
            test_starky(&stark, &config, &generator, &public_inputs)
        );

        // Test the recursive proof.
        // test_recursive_starky(stark, config, generator, &public_inputs);

        timing.print();
    }
}
