use anyhow::Result;

use super::*;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::ec::affine::AffinePoint;
use crate::curta::field::{
    FpDenInstruction, FpInnerProductInstruction, FpMulConstInstruction, FpMulInstruction,
};
use crate::curta::instruction::FromInstructionSet;
use crate::curta::parameters::EdwardsParameters;
use crate::curta::trace::TraceWriter;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Ed25519AddGadget<E: EdwardsParameters> {
    p: AffinePointRegister<E>,
    q: AffinePointRegister<E>,
    pub result: AffinePointRegister<E>,
    x3_numerator_ins: FpInnerProductInstruction<E::BaseField>,
    y3_numerator_ins: FpInnerProductInstruction<E::BaseField>,
    x1_mul_y1_ins: FpMulInstruction<E::BaseField>,
    x2_mul_y2_ins: FpMulInstruction<E::BaseField>,
    f_ins: FpMulInstruction<E::BaseField>,
    d_mul_f_ins: FpMulConstInstruction<E::BaseField>,
    x3_ins: FpDenInstruction<E::BaseField>,
    y3_ins: FpDenInstruction<E::BaseField>,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Adds two elliptic curve points `P` and `Q` on the Ed25519 elliptic curve.
    pub fn ed25519_add<E: EdwardsParameters>(
        &mut self,
        p: &AffinePointRegister<E>,
        q: &AffinePointRegister<E>,
    ) -> Ed25519AddGadget<E>
    where
        L::Instruction: FromInstructionSet<E::BaseField>,
    {
        // Ed25519 Elliptic Curve Addition Formula
        //
        // Given two elliptic curve points (x1, y1) and (x2, y2), compute the sum (x3, y3) with
        //
        // x3 = (x1 * y2 + x2 * y1) / (1 + d * f)
        // y3 = (y1 * y2 + x1 * x2) / (1 - d * f)
        //
        // where f = x1 * x2 * y1 * y2.
        //
        // Reference: https://datatracker.ietf.org/doc/html/draft-josefsson-eddsa-ed25519-02

        let x1 = p.x;
        let x2 = q.x;
        let y1 = p.y;
        let y2 = q.y;

        // x3_numerator = x1 * y2 + x2 * y1.
        let x3_numerator_ins = self.fp_inner_product(&vec![x1, x2], &vec![y2, y1]);

        // y3_numerator = y1 * y2 + x1 * x2.
        let y3_numerator_ins = self.fp_inner_product(&vec![y1, x1], &vec![y2, x2]);

        // f = x1 * x2 * y1 * y2.
        let x1_mul_y1_ins = self.fp_mul(&x1, &y1);
        let x2_mul_y2_ins = self.fp_mul(&x2, &y2);
        let f_ins = self.fp_mul(&x1_mul_y1_ins.result, &x2_mul_y2_ins.result);

        // d * f.
        let d_mul_f_ins = self.fp_mul_const(&f_ins.result, E::D);

        // x3 = x3_numerator / (1 + d * f).
        let x3_ins = self.fp_den(&x3_numerator_ins.result, &d_mul_f_ins.result, true);

        // y3 = y3_numerator / (1 - d * f).
        let y3_ins = self.fp_den(&y3_numerator_ins.result, &d_mul_f_ins.result, false);

        // R = (x3, y3).
        let result = AffinePointRegister::new(x3_ins.result, y3_ins.result);

        Ed25519AddGadget {
            p: *p,
            q: *q,
            result,
            x3_numerator_ins,
            y3_numerator_ins,
            x1_mul_y1_ins,
            x2_mul_y2_ins,
            f_ins,
            d_mul_f_ins,
            x3_ins,
            y3_ins,
        }
    }

    /// Doubles an elliptic curve point `P` on the Ed25519 elliptic curve. Under the hood, the
    /// addition formula is used.
    pub fn ed25519_double<E: EdwardsParameters>(
        &mut self,
        p: &AffinePointRegister<E>,
    ) -> Ed25519AddGadget<E>
    where
        L::Instruction: FromInstructionSet<E::BaseField>,
    {
        self.ed25519_add(p, p)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    pub fn write_ed25519_add<E: EdwardsParameters>(
        &self,
        row_index: usize,
        p: &AffinePoint<E>,
        q: &AffinePoint<E>,
        gadget: Ed25519AddGadget<E>,
    ) -> Result<AffinePoint<E>> {
        let x1 = &p.x;
        let x2 = &q.x;
        let y1 = &p.y;
        let y2 = &q.y;

        // x3_numerator = x1 * y2 + x2 * y1.
        let x3_numerator = self.write_fp_inner_product(
            row_index,
            vec![&x1, &x2],
            vec![&y2, &y1],
            gadget.x3_numerator_ins,
        )?;

        // y3_numerator = y1 * y2 + x1 * x2.
        let y3_numerator = self.write_fp_inner_product(
            row_index,
            vec![&y1, &x1],
            vec![&y2, &x2],
            gadget.y3_numerator_ins,
        )?;

        // f = x1 * x2 * y1 * y2.
        let x1_mul_y1 = self.write_fpmul(row_index, &p.x, &p.y, gadget.x1_mul_y1_ins)?;
        let x2_mul_y2 = self.write_fpmul(row_index, &q.x, &q.y, gadget.x2_mul_y2_ins)?;
        let f = self.write_fpmul(row_index, &x1_mul_y1, &x2_mul_y2, gadget.f_ins)?;

        // d * f.
        let d_mul_f = self.write_fpmul_const(row_index, &f, gadget.d_mul_f_ins)?;

        // x3 = x3_numerator / (1 + d * f).
        let x3 = self.write_fp_den(row_index, &x3_numerator, &d_mul_f, true, gadget.x3_ins)?;

        // y3 = y3_numerator / (1 - d * f).
        let y3 = self.write_fp_den(row_index, &y3_numerator, &d_mul_f, false, gadget.y3_ins)?;

        Ok(AffinePoint::new(x3, y3))
    }

    pub fn write_ed25519_double<E: EdwardsParameters>(
        &self,
        row_index: usize,
        p: &AffinePoint<E>,
        gadget: Ed25519AddGadget<E>,
    ) -> Result<AffinePoint<E>> {
        self.write_ed25519_add(row_index, p, p, gadget)
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
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
    use crate::curta::instruction::InstructionSet;
    use crate::curta::parameters::ed25519::{Ed25519, Ed25519BaseField};
    use crate::curta::trace::trace;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    pub struct Ed25519AddTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for Ed25519AddTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 800;
        const NUM_FREE_COLUMNS: usize = 0;
        type Instruction = InstructionSet<Ed25519BaseField>;
    }

    #[test]
    fn test_ed25519_add() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519;
        type S = TestStark<Ed25519AddTest, F, D>;
        let _ = env_logger::builder().is_test(true).try_init();

        // Build the stark.
        let mut builder = StarkBuilder::<Ed25519AddTest, F, D>::new();
        let p = builder.alloc_local_ec_point::<E>();
        let q = builder.alloc_local_ec_point::<E>();
        let ed25519_add_gadget = builder.ed25519_add::<E>(&p, &q);
        builder.write_ec_point(&p).unwrap();
        builder.write_ec_point(&q).unwrap();
        let (chip, spec) = builder.build();

        // Generate the trace.
        let mut timing = TimingTree::new("ed25519 add", log::Level::Debug);
        let base = E::generator();
        let nb_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);
        let trace = timed!(timing, "witness generation", {
            for i in 0..256 {
                let handle = handle.clone();
                let base = base.clone();
                let ed25519_add_gadget = ed25519_add_gadget.clone();
                rayon::spawn(move || {
                    let mut rng = thread_rng();
                    let a = rng.gen_biguint(256);
                    let b = rng.gen_biguint(256);
                    let p_int = &base * &a;
                    let q_int = &base * &b;
                    let r_exp = &p_int + &q_int;
                    for j in 0..256 {
                        handle.write_ec_point(256 * i + j, &p_int, &p).unwrap();
                        handle.write_ec_point(256 * i + j, &q_int, &q).unwrap();
                        let r = handle
                            .write_ed25519_add(
                                256 * i + j,
                                &p_int,
                                &q_int,
                                ed25519_add_gadget.clone(),
                            )
                            .unwrap();
                        assert_eq!(r, r_exp);
                    }
                })
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

        // Verify recursive proof in a circuit.
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

    #[derive(Clone, Debug, Copy)]
    pub struct EdDoubleTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for EdDoubleTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 768;
        const NUM_FREE_COLUMNS: usize = 0;
        type Instruction = InstructionSet<Ed25519BaseField>;
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_ed_double_row() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519;
        type S = TestStark<EdDoubleTest, F, D>;

        let _ = env_logger::builder().is_test(true).try_init();
        // build the stark
        let mut builder = StarkBuilder::<EdDoubleTest, F, D>::new();

        let P = builder.alloc_ec_point::<E>();

        let ed_double_data = builder.ed25519_double(&P);
        builder.write_ec_point(&P).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let base = E::generator();
        let mut timing = TimingTree::new("Ed_double row", log::Level::Debug);

        let trace = timed!(timing, "generate trace", {
            for i in 0..256usize {
                let handle = handle.clone();
                let base = base.clone();
                let ed_double_data_copy = ed_double_data.clone();
                rayon::spawn(move || {
                    let mut rng = thread_rng();
                    let a = rng.gen_biguint(256);
                    let P_int = &base * &a;
                    let R_exp = &P_int + &P_int;

                    for j in 0..256usize {
                        handle.write_ec_point(256 * i + j, &P_int, &P).unwrap();
                        let R = handle
                            .write_ed25519_double(256 * i + j, &P_int, ed_double_data_copy.clone())
                            .unwrap();
                        assert_eq!(R, R_exp);
                    }
                });
            }
            drop(handle);

            generator.generate_trace(&chip, num_rows as usize).unwrap()
        });

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

        // Verify proof as a stark
        let proof = timed!(
            timing,
            "generate stark proof",
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

        // Verify recursive proof in a circuit
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

        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
