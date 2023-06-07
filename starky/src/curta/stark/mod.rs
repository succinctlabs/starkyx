pub mod config;
pub mod fibonacchi;
pub mod get_challenges;
pub mod proof;
pub mod prover;
pub mod recursive_verifier;
pub mod vars;
pub mod verifier;

use alloc::vec;
use alloc::vec::Vec;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::fri::structure::{
    FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOracleInfo,
    FriPolynomialInfo,
};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::air::parser::AirParser;
use crate::config::StarkConfig;

/// Represents a STARK system.
pub trait Stark<F: RichField + Extendable<D>, const D: usize, const R: usize>: Sync {
    /// The total number of columns
    const COLUMNS: usize;
    /// The number of public inputs.
    const PUBLIC_INPUTS: usize;
    // The total number of challenge elements
    const CHALLENGES: usize;

    /// Columns for each round
    fn round_lengths(&self) -> [usize; R];

    /// The number of challenges per round
    fn num_challenges(&self, round: usize) -> usize;

    fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP);

    /// The maximum constraint degree.
    fn constraint_degree(&self) -> usize;

    /// The maximum constraint degree.
    fn quotient_degree_factor(&self) -> usize {
        1.max(self.constraint_degree() - 1)
    }

    fn num_quotient_polys(&self, config: &StarkConfig) -> usize {
        self.quotient_degree_factor() * config.num_challenges
    }

    /// Computes the FRI instance used to prove this Stark.
    fn fri_instance(
        &self,
        zeta: F::Extension,
        g: F,
        config: &StarkConfig,
    ) -> FriInstanceInfo<F, D> {
        let mut oracles = vec![];
        let mut trace_info: Vec<FriPolynomialInfo> = vec![];

        for length in self.round_lengths() {
            let round_info = FriPolynomialInfo::from_range(oracles.len(), 0..length);
            trace_info.extend(round_info);
            oracles.push(FriOracleInfo {
                num_polys: length,
                blinding: false,
            });
        }

        let num_quotient_polys = self.quotient_degree_factor() * config.num_challenges;
        let quotient_info = FriPolynomialInfo::from_range(oracles.len(), 0..num_quotient_polys);
        oracles.push(FriOracleInfo {
            num_polys: num_quotient_polys,
            blinding: false,
        });

        let zeta_batch = FriBatchInfo {
            point: zeta,
            polynomials: [trace_info.clone(), quotient_info].concat(),
        };
        let zeta_next_batch = FriBatchInfo {
            point: zeta.scalar_mul(g),
            polynomials: trace_info,
        };

        let batches = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfo { oracles, batches }
    }

    /// Computes the FRI instance used to prove this Stark.
    fn fri_instance_target(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
        g: F,
        config: &StarkConfig,
    ) -> FriInstanceInfoTarget<D> {
        let mut oracles = vec![];
        let mut trace_info: Vec<FriPolynomialInfo> = vec![];

        for length in self.round_lengths() {
            let round_info = FriPolynomialInfo::from_range(oracles.len(), 0..length);
            trace_info.extend(round_info);
            oracles.push(FriOracleInfo {
                num_polys: length,
                blinding: false,
            });
        }

        let num_quotient_polys = self.quotient_degree_factor() * config.num_challenges;
        let quotient_info = FriPolynomialInfo::from_range(oracles.len(), 0..num_quotient_polys);
        oracles.push(FriOracleInfo {
            num_polys: num_quotient_polys,
            blinding: false,
        });

        let zeta_batch = FriBatchInfoTarget {
            point: zeta,
            polynomials: [trace_info.clone(), quotient_info].concat(),
        };

        let zeta_next = builder.mul_const_extension(g, zeta);
        let zeta_next_batch = FriBatchInfoTarget {
            point: zeta_next,
            polynomials: trace_info,
        };

        let batches = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfoTarget { oracles, batches }
    }
}
