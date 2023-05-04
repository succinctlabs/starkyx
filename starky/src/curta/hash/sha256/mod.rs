//! This module implements the SHA256 hash function with an input message length of
//! 512 bits as an AIR-based recursive relation.

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::hash::sha256::helper::usize_to_be_bits;
use crate::curta::register::{ArrayRegister, BitRegister, ElementRegister, Register};

pub mod educational;
pub mod helper;
pub mod reference;

// First 32 bits of the fractional parts of the square roots of the first 8 primes.
// Reference: https://en.wikipedia.org/wiki/SHA-2
const H: [usize; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

// First 32 bits of the fractional parts of the cube roots of the first 64 primes.
// Reference: https://en.wikipedia.org/wiki/SHA-2
const K: [usize; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const U64_BIT_LENGTH: usize = 64;
const SHA256_INPUT_LENGTH: usize = 512;
const SHA256_DIGEST_LENGTH: usize = 256;

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    #[allow(non_snake_case)]
    pub fn sha256(&mut self) {
        // The input message of length 512 bits condensed into 16 u32s.
        let input_u32s = self.alloc_array::<ElementRegister>(16);
    }
}

mod tests {
    use num::bigint::RandBigInt;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{StarkParameters, TestStark};
    use crate::curta::constraint::expression::ArithmeticExpression;
    use crate::curta::hash::sha256::helper::{
        and2, be_bits_to_usize, not, rotate, shr, xor2, xor3,
    };
    use crate::curta::instruction::InstructionSet;
    use crate::curta::parameters::ed25519::{Ed25519, Ed25519BaseField};
    use crate::curta::register::{RegisterSerializable, U16Register};
    use crate::curta::trace::trace;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    pub struct Sha256Test;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for Sha256Test {
        const NUM_ARITHMETIC_COLUMNS: usize = 4;
        const NUM_FREE_COLUMNS: usize = 900;
        type Instruction = InstructionSet<Ed25519BaseField>;
    }

    struct Sha256Gadget {
        pc: ArrayRegister<BitRegister>,
        input: ArrayRegister<ElementRegister>,
        w: ArrayRegister<ElementRegister>,
        w_j_minus_15_bits: ArrayRegister<BitRegister>,
        w_j_minus_2_bits: ArrayRegister<BitRegister>,
        mixing_s0_witness: ArrayRegister<BitRegister>,
        mixing_s0: ArrayRegister<BitRegister>,
        mixing_s1_witness: ArrayRegister<BitRegister>,
        mixing_s1: ArrayRegister<BitRegister>,
        w_j_minus_16: ElementRegister,
        w_j_minus_7: ElementRegister,
        mixing_add4_carry: ElementRegister,
        mixing_add4_result_low: U16Register,
        mixing_add4_result_high: U16Register,
        h: ArrayRegister<ElementRegister>,
        compress_sa_bits: ArrayRegister<BitRegister>,
        compress_sb_bits: ArrayRegister<BitRegister>,
        compress_sc_bits: ArrayRegister<BitRegister>,
        compress_sd_bits: ArrayRegister<BitRegister>,
        compress_se_bits: ArrayRegister<BitRegister>,
        compress_sf_bits: ArrayRegister<BitRegister>,
        compress_sg_bits: ArrayRegister<BitRegister>,
        compress_sh_bits: ArrayRegister<BitRegister>,
        compress_s1_witness: ArrayRegister<BitRegister>,
        compress_s1: ArrayRegister<BitRegister>,
        compress_ch_witness_1: ArrayRegister<BitRegister>,
        compress_ch_witness_2: ArrayRegister<BitRegister>,
        compress_ch: ArrayRegister<BitRegister>,
        compress_temp1_carry: ElementRegister,
        compress_temp1_witness_low: U16Register,
        compress_temp1_witness_high: U16Register,
    }

    impl Sha256Gadget {
        fn new<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize>(
            builder: &mut StarkBuilder<L, F, D>,
        ) -> Self {
            // The program counter.
            let pc = builder.alloc_array::<BitRegister>(NB_STEPS);

            // Input registers.
            let input = builder.alloc_array::<ElementRegister>(16);
            let w = builder.alloc_array::<ElementRegister>(64);

            // Compute s0 & s1.
            let w_j_minus_15_bits = builder.alloc_array::<BitRegister>(32);
            let w_j_minus_2_bits = builder.alloc_array::<BitRegister>(32);
            let s0_witness = builder.alloc_array::<BitRegister>(32);
            let s0 = builder.alloc_array::<BitRegister>(32);
            let s1_witness = builder.alloc_array::<BitRegister>(32);
            let s1 = builder.alloc_array::<BitRegister>(32);

            // Mix.
            let w_j_minus_16 = builder.alloc::<ElementRegister>();
            let w_j_minus_7 = builder.alloc::<ElementRegister>();
            let carry = builder.alloc::<ElementRegister>();
            let witness_low = builder.alloc::<U16Register>();
            let witness_high = builder.alloc::<U16Register>();

            // Compress.
            let h = builder.alloc_array::<ElementRegister>(8);
            let compress_sa_bits = builder.alloc_array::<BitRegister>(32);
            let compress_sb_bits = builder.alloc_array::<BitRegister>(32);
            let compress_sc_bits = builder.alloc_array::<BitRegister>(32);
            let compress_sd_bits = builder.alloc_array::<BitRegister>(32);
            let compress_se_bits = builder.alloc_array::<BitRegister>(32);
            let compress_sf_bits = builder.alloc_array::<BitRegister>(32);
            let compress_sg_bits = builder.alloc_array::<BitRegister>(32);
            let compress_sh_bits = builder.alloc_array::<BitRegister>(32);
            let compress_s1_witness = builder.alloc_array::<BitRegister>(32);
            let compress_s1 = builder.alloc_array::<BitRegister>(32);
            let compress_ch_witness_1 = builder.alloc_array::<BitRegister>(32);
            let compress_ch_witness_2 = builder.alloc_array::<BitRegister>(32);
            let compress_ch = builder.alloc_array::<BitRegister>(32);
            let compress_temp1_carry = builder.alloc::<ElementRegister>();
            let compress_temp1_witness_low = builder.alloc::<U16Register>();
            let compress_temp1_witness_high = builder.alloc::<U16Register>();

            let gadget = Self {
                pc,
                input,
                w,
                w_j_minus_15_bits,
                w_j_minus_2_bits,
                mixing_s0_witness: s0_witness,
                mixing_s0: s0,
                mixing_s1_witness: s1_witness,
                mixing_s1: s1,
                w_j_minus_16,
                w_j_minus_7,
                mixing_add4_carry: carry,
                mixing_add4_result_low: witness_low,
                mixing_add4_result_high: witness_high,
                h,
                compress_sa_bits,
                compress_sb_bits,
                compress_sc_bits,
                compress_sd_bits,
                compress_se_bits,
                compress_sf_bits,
                compress_sg_bits,
                compress_sh_bits,
                compress_s1_witness,
                compress_s1,
                compress_ch_witness_1,
                compress_ch_witness_2,
                compress_ch,
                compress_temp1_carry,
                compress_temp1_witness_low,
                compress_temp1_witness_high,
            };

            gadget.constraints(builder);
            gadget
        }

        fn xor2<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize>(
            &self,
            a: ArithmeticExpression<F, D>,
            b: ArithmeticExpression<F, D>,
            c: ArithmeticExpression<F, D>,
        ) -> ArithmeticExpression<F, D> {
            c - (a.clone() + b.clone() - a * b * F::TWO)
        }

        fn add<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize>(
            &self,
            x: Vec<ArithmeticExpression<F, D>>,
            carry: ArithmeticExpression<F, D>,
            witness_low: ArithmeticExpression<F, D>,
            witness_high: ArithmeticExpression<F, D>,
        ) -> ArithmeticExpression<F, D> {
            let limb = F::from_canonical_u64(1 << 16);
            let modulus = F::from_canonical_u64(1 << 32);
            let sum = x
                .into_iter()
                .fold(ArithmeticExpression::from_constant(F::ZERO), |acc, x| {
                    acc + x
                });
            sum - carry * modulus - witness_low - witness_high * limb
        }

        fn u32_witnesses_to_element<
            L: StarkParameters<F, D>,
            F: RichField + Extendable<D>,
            const D: usize,
        >(
            &self,
            x: U16Register,
            y: U16Register,
        ) -> ArithmeticExpression<F, D> {
            let limb = F::from_canonical_u64(1 << 16);
            x.expr() + y.expr() * limb
        }

        /// Constrain pc.sum() == 1 and pc[i % NB_STEPS] == pc[(i + 1) % NB_STEPS].next().
        fn pc_constraints<
            L: StarkParameters<F, D>,
            F: RichField + Extendable<D>,
            const D: usize,
        >(
            &self,
            builder: &mut StarkBuilder<L, F, D>,
        ) {
            builder.constrain(self.pc.expr().sum() - F::ONE);
            for i in 0..NB_STEPS {
                builder.constrain_transition(
                    self.pc.get(i % NB_STEPS).expr()
                        - self.pc.get((i + 1) % NB_STEPS).next().expr(),
                )
            }
        }

        fn mixing_constraints<
            L: StarkParameters<F, D>,
            F: RichField + Extendable<D>,
            const D: usize,
        >(
            &self,
            builder: &mut StarkBuilder<L, F, D>,
        ) {
            // Constrain w[j-15].bits(), w[j-2].bits(), and w[j-16], w[j-7] for j in 16..64.
            for j in 16..64 {
                let step = j - 16;
                builder.constrain(
                    self.pc.get(step).expr()
                        * (self.w_j_minus_15_bits.expr().be_sum() - self.w.get(j - 15).expr()),
                );
                builder.constrain(
                    self.pc.get(step).expr()
                        * (self.w_j_minus_2_bits.expr().be_sum() - self.w.get(j - 2).expr()),
                );
                builder.constrain(
                    self.pc.get(step).expr()
                        * (self.w_j_minus_16.expr() - self.w.get(j - 16).expr()),
                );
                builder.constrain(
                    self.pc.get(step).expr() * (self.w_j_minus_7.expr() - self.w.get(j - 7).expr()),
                );
            }

            // Compute s0 & s1.
            let w_j_minus_15_rotated_7_bits = self.w_j_minus_15_bits.expr::<F, D>().rotate(7);
            let w_j_minus_15_rotated_18_bits = self.w_j_minus_15_bits.expr::<F, D>().rotate(18);
            let w_j_minus_15_shifted_3_bits = self.w_j_minus_15_bits.expr::<F, D>().shr(3);
            builder.constrain(self.xor2::<L, F, D>(
                w_j_minus_15_rotated_7_bits,
                w_j_minus_15_rotated_18_bits,
                self.mixing_s0_witness.expr(),
            ));
            builder.constrain(self.xor2::<L, F, D>(
                self.mixing_s0_witness.expr(),
                w_j_minus_15_shifted_3_bits,
                self.mixing_s0.expr(),
            ));
            builder.constrain(self.xor2::<L, F, D>(
                self.w_j_minus_2_bits.expr::<F, D>().rotate(17),
                self.w_j_minus_2_bits.expr::<F, D>().rotate(19),
                self.mixing_s1_witness.expr(),
            ));
            builder.constrain(self.xor2::<L, F, D>(
                self.mixing_s1_witness.expr(),
                self.w_j_minus_2_bits.expr::<F, D>().shr(10),
                self.mixing_s1.expr(),
            ));

            // w[j] = add4(w[j - 16], s0, w[j - 7], s1) is equivalent to
            // w[j-16] + s0 + w[j-7] + s1 - carry_bit * 2^32 - witness_low - witness_high * 2^16 == 0
            builder.constrain(self.add::<L, F, D>(
                vec![
                    self.w_j_minus_16.expr(),
                    self.mixing_s0.expr().be_sum(),
                    self.w_j_minus_7.expr(),
                    self.mixing_s1.expr().be_sum(),
                ],
                self.mixing_add4_carry.expr(),
                self.mixing_add4_result_low.expr(),
                self.mixing_add4_result_high.expr(),
            ));

            // For each mixing round, result = add4(w[j - 16], s0, w[j - 7], s1) and
            // w[i].next() = \sum_{i\neqj} step(i) * w[i] + step(j) * result
            let add4_result = self.u32_witnesses_to_element::<L, F, D>(
                self.mixing_add4_result_low,
                self.mixing_add4_result_high,
            );
            for j in 16..64 {
                let step = j - 16;
                let w_j_next = self.w.get(j).next().expr();
                let mut root = self.pc.get(step).expr() * (w_j_next.clone() - add4_result.clone());
                for k in 16..64 {
                    let pseudo_step = k - 16;
                    if pseudo_step != step {
                        root = root
                            + self.pc.get(pseudo_step).expr()
                                * (w_j_next.clone() - self.w.get(j).expr());
                    }
                }
                builder.constrain_transition(root);
            }
        }

        fn compressing_constraints<
            L: StarkParameters<F, D>,
            F: RichField + Extendable<D>,
            const D: usize,
        >(
            &self,
            builder: &mut StarkBuilder<L, F, D>,
        ) {
            let sa = self.h.get(0);
            let sb = self.h.get(1);
            let sc = self.h.get(2);
            let sd = self.h.get(3);
            let se = self.h.get(4);
            let sf = self.h.get(5);
            let sg = self.h.get(6);
            let sh = self.h.get(7);

            // COMPUTE S1

            // decompose se to bits
            builder.constrain(self.compress_sa_bits.expr().be_sum() - sa.expr());
            builder.constrain(self.compress_sb_bits.expr().be_sum() - sb.expr());
            builder.constrain(self.compress_sc_bits.expr().be_sum() - sc.expr());
            builder.constrain(self.compress_sd_bits.expr().be_sum() - sd.expr());
            builder.constrain(self.compress_se_bits.expr().be_sum() - se.expr());
            builder.constrain(self.compress_sf_bits.expr().be_sum() - sf.expr());
            builder.constrain(self.compress_sg_bits.expr().be_sum() - sg.expr());
            builder.constrain(self.compress_sh_bits.expr().be_sum() - sh.expr());

            // s1 = xor3(rotate(se, 6), rotate(se, 11), rotate(se, 25))
            builder.constrain(self.xor2::<L, F, D>(
                self.compress_se_bits.expr().rotate(6),
                self.compress_se_bits.expr().rotate(11),
                self.compress_s1_witness.expr(),
            ));
            builder.constrain(self.xor2::<L, F, D>(
                self.compress_s1_witness.expr(),
                self.compress_se_bits.expr().rotate(25),
                self.compress_s1.expr(),
            ));

            // COMPUTE CH
            // ch = xor2(and2(se, sf), and2(not(se), sg))
            builder.constrain(
                self.compress_ch_witness_1.expr()
                    - self.compress_se_bits.expr() * self.compress_sf_bits.expr(),
            );
            builder.constrain(
                self.compress_ch_witness_2.expr()
                    - (ArithmeticExpression::from_constant_vec([F::ONE; 32].to_vec())
                        - self.compress_se_bits.expr())
                        * self.compress_sg_bits.expr(),
            );
            builder.constrain(self.xor2::<L, F, D>(
                self.compress_ch_witness_1.expr(),
                self.compress_ch_witness_2.expr(),
                self.compress_ch.expr(),
            ));

            // COMPUTE TEMP1

            let mut w_j = ArithmeticExpression::from_constant(F::ZERO);
            let mut k_j = ArithmeticExpression::from_constant(F::ZERO);
            for j in 0..64 {
                w_j = w_j + self.pc.get(48 + j).expr() * self.w.get(j).expr();
                k_j = k_j + self.pc.get(48 + j).expr() * F::from_canonical_u64(K[j] as u64);
            }

            builder.constrain(self.add::<L, F, D>(
                vec![
                    self.compress_sh_bits.expr().be_sum(),
                    self.compress_s1.expr().be_sum(),
                    self.compress_ch.expr().be_sum(),
                    k_j,
                    w_j,
                ],
                self.compress_temp1_carry.expr(),
                self.compress_temp1_witness_low.expr(),
                self.compress_temp1_witness_high.expr(),
            ));
        }

        fn constraints<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize>(
            &self,
            builder: &mut StarkBuilder<L, F, D>,
        ) {
            self.pc_constraints(builder);

            // The input message of length 512 bits condensed into 16 u32s.
            for j in 0..16 {
                builder.constrain(
                    self.pc.get(0).expr() * (self.w.get(j).expr() - self.input.get(j).expr()),
                );
            }

            self.mixing_constraints(builder);
            self.compressing_constraints(builder);
        }
    }

    const NB_MIXING_STEPS: usize = 48;
    const NB_COMPRESS_STEPS: usize = 64;
    const NB_ACCUMULATE_STEPS: usize = 1;
    const NB_STEPS: usize = NB_MIXING_STEPS + NB_COMPRESS_STEPS + NB_ACCUMULATE_STEPS;

    #[test]
    fn test_sha256() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519;
        type S = TestStark<Sha256Test, F, D>;
        let _ = env_logger::builder().is_test(true).try_init();

        // Build the stark.
        let mut stark = StarkBuilder::<Sha256Test, F, D>::new();

        let gadget = Sha256Gadget::new(&mut stark);
        // gadget.pc_constraints(&mut stark);

        let (chip, spec) = stark.build();

        // Generate the trace.
        let mut timing = TimingTree::new("sha256", log::Level::Debug);
        let nb_rows = 2usize.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let mut input_gt = [0u32; 16];
        for i in 0..16 {
            input_gt[i] = i as u32;
        }

        let mut wiggle = [0u32; 64];
        for i in 0..16 {
            wiggle[i] = input_gt[i];
        }

        fn u32_arr_to_field_array(x: &[u32]) -> Vec<F> {
            x.into_iter()
                .map(|x| F::from_canonical_u32(*x))
                .collect::<Vec<_>>()
        }

        fn u32_to_be_field_bits(x: u32) -> Vec<F> {
            usize_to_be_bits::<32>(x as usize)
                .into_iter()
                .map(|x| F::from_canonical_u32(x as u32))
                .collect::<Vec<_>>()
        }

        fn bits_to_field_bits<const L: usize>(x: [bool; L]) -> Vec<F> {
            x.into_iter()
                .map(|x| F::from_canonical_u32(x as u32))
                .collect::<Vec<_>>()
        }

        const SHA256_WORD_LENGTH: usize = 32;
        let mut h = H
            .into_iter()
            .map(|x| usize_to_be_bits::<SHA256_WORD_LENGTH>(x))
            .collect::<Vec<_>>();

        let trace = timed!(timing, "witness generation", {
            for i in 0..nb_rows {
                // Write input.
                handle.write_data_v2(i, gadget.input, u32_arr_to_field_array(&input_gt));

                // Write step selectors.
                let step = i % NB_STEPS;
                for j in 0..NB_STEPS {
                    handle.write_bit(i, j == step, &gadget.pc.get(j))
                }

                if step == 0 {
                    for i in 0..16 {
                        wiggle[i] = input_gt[i];
                    }
                    for i in 16..64 {
                        wiggle[i] = 0;
                    }
                    h = H
                        .into_iter()
                        .map(|x| usize_to_be_bits::<SHA256_WORD_LENGTH>(x))
                        .collect::<Vec<_>>();
                }

                // Write w.
                handle.write_data_v2(i, gadget.w, u32_arr_to_field_array(&wiggle));

                let is_mixing = step < NB_MIXING_STEPS;
                if is_mixing {
                    let j = 16 + step;
                    let w15 = u32_to_be_field_bits(wiggle[j - 15]);
                    handle.write_data_v2(i, gadget.w_j_minus_15_bits, w15);
                    let w2 = u32_to_be_field_bits(wiggle[j - 2]);
                    handle.write_data_v2(i, gadget.w_j_minus_2_bits, w2);

                    let w_j_minus_15_val = usize_to_be_bits::<32>(wiggle[j - 15] as usize);
                    let s0_witness_val =
                        xor2(rotate(w_j_minus_15_val, 7), rotate(w_j_minus_15_val, 18));
                    handle.write_data_v2(
                        i,
                        gadget.mixing_s0_witness,
                        bits_to_field_bits(s0_witness_val),
                    );

                    let s0_value = xor2(s0_witness_val, shr(w_j_minus_15_val, 3));
                    handle.write_data_v2(i, gadget.mixing_s0, bits_to_field_bits(s0_value));

                    let w_j_minus_2_val = usize_to_be_bits::<32>(wiggle[j - 2] as usize);
                    let s1_witness_val =
                        xor2(rotate(w_j_minus_2_val, 17), rotate(w_j_minus_2_val, 19));
                    handle.write_data_v2(
                        i,
                        gadget.mixing_s1_witness,
                        bits_to_field_bits(s1_witness_val),
                    );

                    let s1_value = xor2(s1_witness_val, shr(w_j_minus_2_val, 10));
                    handle.write_data_v2(i, gadget.mixing_s1, bits_to_field_bits(s1_value));

                    let w16 = wiggle[j - 16];
                    handle.write_data_v2(i, gadget.w_j_minus_16, vec![F::from_canonical_u32(w16)]);
                    let w7 = wiggle[j - 7];
                    handle.write_data_v2(i, gadget.w_j_minus_7, vec![F::from_canonical_u32(w7)]);

                    let sum = be_bits_to_usize(usize_to_be_bits::<32>(wiggle[j - 16] as usize))
                        + be_bits_to_usize(s0_value)
                        + be_bits_to_usize(usize_to_be_bits::<32>(wiggle[j - 7] as usize))
                        + be_bits_to_usize(s1_value);
                    let carry_val = sum / (1 << 32);
                    handle.write_data_v2(
                        i,
                        gadget.mixing_add4_carry,
                        vec![F::from_canonical_u64(carry_val as u64)],
                    );
                    let reduced_sum = sum - (carry_val as usize) * (1 << 32);

                    // Decompose reduced_sum into two limbs, where the base is 2^16.
                    let witness_low_value = F::from_canonical_u64((reduced_sum % (1 << 16)) as u64);
                    let witness_high_value = F::from_canonical_u64((reduced_sum >> 16) as u64);
                    assert!(witness_high_value.to_canonical_u64() < (1 << 16));
                    assert!(
                        witness_low_value + witness_high_value * F::from_canonical_u64(1 << 16)
                            == F::from_canonical_u64(reduced_sum as u64)
                    );
                    assert!(witness_low_value.to_canonical_u64() < (1 << 16));
                    handle.write_data_v2(i, gadget.mixing_add4_result_low, vec![witness_low_value]);
                    handle.write_data_v2(
                        i,
                        gadget.mixing_add4_result_high,
                        vec![witness_high_value],
                    );

                    assert!(
                        (sum as usize)
                            - (carry_val as usize) * (1 << 32)
                            - ((reduced_sum as usize) % (1 << 16))
                            - ((reduced_sum as usize) >> 16) * (1 << 16)
                            == 0
                    );

                    wiggle[j] = (((reduced_sum as usize) % (1 << 16))
                        + ((reduced_sum as usize) >> 16) * (1 << 16))
                        as u32;
                }

                handle.write_data_v2(
                    i,
                    gadget.h,
                    h.clone()
                        .into_iter()
                        .map(|x| F::from_canonical_u64(be_bits_to_usize(x) as u64))
                        .collect(),
                );

                let sa = h[0];
                let sb = h[1];
                let sc = h[2];
                let sd = h[3];
                let se = h[4];
                let sf = h[5];
                let sg = h[6];
                let sh = h[7];
                let sa_field_bits = bits_to_field_bits(sa);
                let sb_field_bits = bits_to_field_bits(sb);
                let sc_field_bits = bits_to_field_bits(sc);
                let sd_field_bits = bits_to_field_bits(sd);
                let se_field_bits = bits_to_field_bits(se);
                let sf_field_bits = bits_to_field_bits(sf);
                let sg_field_bits = bits_to_field_bits(sg);
                let sh_field_bits = bits_to_field_bits(sh);
                handle.write_data_v2(i, gadget.compress_sa_bits, sa_field_bits);
                handle.write_data_v2(i, gadget.compress_sb_bits, sb_field_bits);
                handle.write_data_v2(i, gadget.compress_sc_bits, sc_field_bits);
                handle.write_data_v2(i, gadget.compress_sd_bits, sd_field_bits);
                handle.write_data_v2(i, gadget.compress_se_bits, se_field_bits);
                handle.write_data_v2(i, gadget.compress_sf_bits, sf_field_bits);
                handle.write_data_v2(i, gadget.compress_sg_bits, sg_field_bits);
                handle.write_data_v2(i, gadget.compress_sh_bits, sh_field_bits);

                let compressing_s1_witness_bits =
                    bits_to_field_bits(xor2(rotate(se, 6), rotate(se, 11)));
                handle.write_data_v2(i, gadget.compress_s1_witness, compressing_s1_witness_bits);

                let s1 = xor3(rotate(se, 6), rotate(se, 11), rotate(se, 25));
                let compressing_s1_bits = bits_to_field_bits(s1);
                handle.write_data_v2(i, gadget.compress_s1, compressing_s1_bits);

                let compressing_ch_witness_1 = and2(se, sf);
                let compressing_ch_witness_2 = and2(not(se), sg);
                let compressing_ch = xor2(and2(se, sf), and2(not(se), sg));
                handle.write_data_v2(
                    i,
                    gadget.compress_ch_witness_1,
                    bits_to_field_bits(compressing_ch_witness_1),
                );
                handle.write_data_v2(
                    i,
                    gadget.compress_ch_witness_2,
                    bits_to_field_bits(compressing_ch_witness_2),
                );
                handle.write_data_v2(i, gadget.compress_ch, bits_to_field_bits(compressing_ch));

                let mut sum: usize = 0;
                if NB_MIXING_STEPS <= step && step < NB_MIXING_STEPS + NB_COMPRESS_STEPS {
                    let j = step - 48;
                    sum = be_bits_to_usize(sh)
                        + be_bits_to_usize(s1)
                        + be_bits_to_usize(compressing_ch)
                        + K[j]
                        + (wiggle[j] as usize);
                } else {
                    sum = be_bits_to_usize(sh)
                        + be_bits_to_usize(s1)
                        + be_bits_to_usize(compressing_ch);
                }

                let carry_val = sum / (1 << 32);
                handle.write_data_v2(
                    i,
                    gadget.compress_temp1_carry,
                    vec![F::from_canonical_u64(carry_val as u64)],
                );
                let reduced_sum = sum - (carry_val as usize) * (1 << 32);

                // Decompose reduced_sum into two limbs, where the base is 2^16.
                let witness_low_value = F::from_canonical_u64((reduced_sum % (1 << 16)) as u64);
                let witness_high_value = F::from_canonical_u64((reduced_sum >> 16) as u64);
                assert!(witness_high_value.to_canonical_u64() < (1 << 16));
                assert!(
                    witness_low_value + witness_high_value * F::from_canonical_u64(1 << 16)
                        == F::from_canonical_u64(reduced_sum as u64)
                );
                assert!(witness_low_value.to_canonical_u64() < (1 << 16));
                handle.write_data_v2(
                    i,
                    gadget.compress_temp1_witness_low,
                    vec![witness_low_value],
                );
                handle.write_data_v2(
                    i,
                    gadget.compress_temp1_witness_high,
                    vec![witness_high_value],
                );
            }
            drop(handle);
            generator.generate_trace(&chip, nb_rows as usize).unwrap()
        });

        // Verify proof as a stark
        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);
        let proof = timed!(
            timing,
            "proof generation",
            prove::<F, C, S, D>(
                stark.clone(),
                &config,
                trace,
                [],
                &mut TimingTree::default(),
            )
            .unwrap()
        );
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // // Verify recursive proof in a circuit.
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);
        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof = add_virtual_stark_proof_with_pis(
            &mut recursive_builder,
            stark.clone(),
            &config,
            degree_bits,
        );
        recursive_builder.print_gate_counts(0);
        let mut rec_pw = PartialWitness::new();
        set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);
        verify_stark_proof_circuit::<F, C, S, D>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );
        let recursive_data = recursive_builder.build::<C>();
        let recursive_proof = timed!(
            timing,
            "generate recursive proof",
            plonky2::plonk::prover::prove(
                &recursive_data.prover_only,
                &recursive_data.common,
                rec_pw,
                &mut TimingTree::default(),
            )
            .unwrap()
        );
        recursive_data.verify(recursive_proof).unwrap();
        timing.print();
    }
}
