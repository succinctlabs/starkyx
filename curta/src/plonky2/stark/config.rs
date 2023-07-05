use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::packed::PackedField;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::fri::{FriConfig, FriParams};
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::log2_strict;

use crate::plonky2::challenger::{Plonky2Challenger, Plonky2RecursiveChallenger};
use crate::plonky2::parser::{RecursiveStarkParser, StarkParser};
use crate::stark::config::StarkConfig;

#[derive(Debug, Clone)]
pub struct StarkyConfig<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    pub security_bits: usize,

    /// The number of challenge points to generate, for IOPs that have soundness errors of (roughly)
    /// `degree / |F|`.
    pub num_challenges: usize,

    /// The number of bits in the degree of the column polynomials
    ///
    /// Number Rows = 2^{degree_bits}
    pub degree_bits: usize,

    pub fri_config: FriConfig,

    _marker: core::marker::PhantomData<(F, C)>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    StarkyConfig<F, C, D>
{
    /// A typical configuration with a rate of 2, resulting in fast but large proofs.
    /// Targets ~100 bit conjectured security.
    pub fn standard_fast_config(num_rows: usize) -> Self {
        let degree_bits = log2_strict(num_rows);
        Self {
            security_bits: 100,
            num_challenges: 2,
            degree_bits,
            fri_config: FriConfig {
                rate_bits: 1,
                cap_height: 4,
                proof_of_work_bits: 16,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
                num_query_rounds: 84,
            },
            _marker: core::marker::PhantomData,
        }
    }

    pub(crate) fn fri_params(&self) -> FriParams {
        self.fri_config.fri_params(self.degree_bits, false)
    }
}

impl<'a, F, C: GenericConfig<D, F = F>, FE, P, const D: usize, const D2: usize>
    StarkConfig<StarkParser<'a, F, FE, P, D, D2>> for StarkyConfig<F, C, D>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    type Challenger = Plonky2Challenger<F, C::Hasher>;

    type Proof = ();
}

impl<'a, F, C: GenericConfig<D, F = F>, const D: usize> StarkConfig<RecursiveStarkParser<'a, F, D>>
    for StarkyConfig<F, C, D>
where
    F: RichField + Extendable<D>,
{
    type Challenger = Plonky2RecursiveChallenger<F, C::InnerHasher, D>;

    type Proof = ();
}

pub type PoseidonGoldilocksStarkConfig = StarkyConfig<GoldilocksField, PoseidonGoldilocksConfig, 2>;
