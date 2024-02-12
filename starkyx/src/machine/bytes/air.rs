use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;
use plonky2::util::timing::TimingTree;
use serde::{Deserialize, Serialize};

use crate::air::{RAir, RAirData, RoundDatum};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::operations::NUM_BIT_OPPS;
use crate::chip::uint::operations::instruction::UintInstruction;
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::Starky;
use crate::prelude::AirParser;
use crate::trace::AirTrace;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ByteParameters<F, E>(pub PhantomData<(F, E)>);

impl<F: PrimeField64, E: CubicParameters<F>> AirParameters for ByteParameters<F, E> {
    type Field = F;
    type CubicParams = E;

    type Instruction = UintInstruction;

    const NUM_ARITHMETIC_COLUMNS: usize = 0;
    const NUM_FREE_COLUMNS: usize = 15;
    const EXTENDED_COLUMNS: usize = 45;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ByteAir<F: PrimeField64, E: CubicParameters<F>>(pub(crate) Chip<ByteParameters<F, E>>);

impl<F: PrimeField64, E: CubicParameters<F>> RAirData for ByteAir<F, E> {
    fn width(&self) -> usize {
        self.0.width()
    }

    fn constraint_degree(&self) -> usize {
        self.0.constraint_degree()
    }

    fn round_data(&self) -> Vec<RoundDatum> {
        let total = ByteParameters::<F, E>::num_columns();
        let execution_trace_length = self.0.execution_trace_length;
        let extended_trace_length = total - execution_trace_length;

        vec![
            RoundDatum::new(NUM_BIT_OPPS + 1, (0, 0), 0),
            RoundDatum::new(
                execution_trace_length - (NUM_BIT_OPPS + 1),
                (0, 0),
                self.0.num_challenges,
            ),
            RoundDatum::new(extended_trace_length, (0, self.0.num_global_values), 0),
        ]
    }

    fn num_public_inputs(&self) -> usize {
        self.0.num_public_inputs()
    }
}

impl<AP: AirParser, F: PrimeField64, E: CubicParameters<F>> RAir<AP> for ByteAir<F, E>
where
    Chip<ByteParameters<F, E>>: RAir<AP>,
{
    fn eval(&self, parser: &mut AP) {
        self.0.eval(parser)
    }

    fn eval_global(&self, parser: &mut AP) {
        self.0.eval_global(parser)
    }
}

pub fn get_preprocessed_byte_trace<F, E, C, const D: usize>(
    lookup_writer: &TraceWriter<F>,
    lookup_config: &StarkyConfig<C, D>,
    lookup_stark: &Starky<ByteAir<F, E>>,
) -> PolynomialBatch<F, C::GenericConfig, D>
where
    F: RichField + Extendable<D>,
    E: CubicParameters<F>,
    C: CurtaConfig<D, F = F>,
{
    let lookup_execution_trace_values = lookup_writer
        .read_trace()
        .unwrap()
        .rows_par()
        .flat_map(|row| row[(NUM_BIT_OPPS + 1)..lookup_stark.air.0.execution_trace_length].to_vec())
        .collect::<Vec<_>>();

    let lookup_execution_trace = AirTrace {
        values: lookup_execution_trace_values,
        width: (lookup_stark.air.0.execution_trace_length - (NUM_BIT_OPPS + 1)),
    };

    lookup_config.commit(&lookup_execution_trace, &mut TimingTree::default())
}
