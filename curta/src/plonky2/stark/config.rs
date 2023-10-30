use core::fmt::Debug;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::polynomial::PolynomialValues;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::fri::{FriConfig, FriParams};
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::log2_strict;
use plonky2::util::timing::TimingTree;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::maybe_rayon::*;
use crate::trace::AirTrace;
use crate::utils::serde::{deserialize_fri_config, serialize_fri_config};

pub trait CurtaConfig<const D: usize>:
    Debug + Clone + 'static + Send + Sync + Serialize + DeserializeOwned
{
    type F: RichField + Extendable<D>;
    type FE: FieldExtension<D, BaseField = Self::F>;
    type Hasher: AlgebraicHasher<Self::F>;
    type InnerHasher: AlgebraicHasher<Self::F>;
    type GenericConfig: GenericConfig<
        D,
        F = Self::F,
        FE = Self::FE,
        Hasher = Self::Hasher,
        InnerHasher = Self::InnerHasher,
    >;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkyConfig<C, const D: usize> {
    pub security_bits: usize,

    /// The number of challenge points to generate, for IOPs that have soundness errors of (roughly)
    /// `degree / |F|`.
    pub num_challenges: usize,

    /// The number of bits in the degree of the column polynomials
    ///
    /// Number Rows = 2^{degree_bits}
    pub degree_bits: usize,

    #[serde(serialize_with = "serialize_fri_config")]
    #[serde(deserialize_with = "deserialize_fri_config")]
    pub fri_config: FriConfig,

    _marker: core::marker::PhantomData<C>,
}

impl<C: CurtaConfig<D>, const D: usize> StarkyConfig<C, D> {
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

    pub fn fri_params(&self) -> FriParams {
        self.fri_config.fri_params(self.degree_bits, false)
    }

    pub fn commit(
        &self,
        trace: &AirTrace<C::F>,
        timing: &mut TimingTree,
    ) -> PolynomialBatch<C::F, C::GenericConfig, D> {
        let trace_cols = trace
            .as_columns()
            .into_par_iter()
            .map(PolynomialValues::from)
            .collect::<Vec<_>>();

        let rate_bits = self.fri_config.rate_bits;
        let cap_height = self.fri_config.cap_height;
        PolynomialBatch::<C::F, C::GenericConfig, D>::from_values(
            trace_cols, rate_bits, false, cap_height, timing, None,
        )
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CurtaPoseidonGoldilocksConfig;

impl CurtaConfig<2> for CurtaPoseidonGoldilocksConfig {
    type F = <PoseidonGoldilocksConfig as GenericConfig<2>>::F;
    type FE = <PoseidonGoldilocksConfig as GenericConfig<2>>::FE;
    type Hasher = <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher;
    type InnerHasher = <PoseidonGoldilocksConfig as GenericConfig<2>>::InnerHasher;
    type GenericConfig = PoseidonGoldilocksConfig;
}

pub type PoseidonGoldilocksStarkConfig = StarkyConfig<CurtaPoseidonGoldilocksConfig, 2>;
