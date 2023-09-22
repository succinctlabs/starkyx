//! This module implements a lookup argument based on the logarithmic derivative as in
//! https://eprint.iacr.org/2022/1530.pdf
//!

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::chip::builder::AirBuilder;
use crate::chip::constraint::Constraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::Register;
use crate::chip::table::lookup::log_der::constraint::LookupConstraint;
use crate::chip::table::lookup::Lookup;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub mod constraint;
pub mod trace;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupTable<T: Register, F: Field, E: CubicParameters<F>> {
    pub(crate) challenge: CubicRegister,
    pub(crate) table: Vec<T>,
    pub(crate) multiplicities: ArrayRegister<ElementRegister>,
    pub(crate) multiplicities_table_log: ArrayRegister<CubicRegister>,
    pub(crate) table_accumulator: CubicRegister,
    pub(crate) digest: CubicRegister,
    _marker: core::marker::PhantomData<(F, E)>,
}

/// Currently, only supports an even number of values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogLookupValues<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    pub(crate) challenge: CubicRegister,
    pub(crate) trace_values: Vec<T>,
    pub(crate) public_values: Vec<T>,
    pub(crate) row_accumulators: ArrayRegister<CubicRegister>,
    pub(crate) global_accumulators: ArrayRegister<CubicRegister>,
    pub(crate) log_lookup_accumulator: CubicRegister,
    pub local_digest: CubicRegister,
    pub global_digest: Option<CubicRegister>,
    pub digest: CubicRegister,
    _marker: core::marker::PhantomData<(F, E)>,
}

/// Currently, only supports an even number of values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogLookup<T: EvalCubic, F: Field, E: CubicParameters<F>> {
    pub(crate) table_data: LookupTable<T, F, E>,
    pub(crate) values_data: LogLookupValues<T, F, E>,
    _marker: core::marker::PhantomData<(F, E)>,
}

// LogLookUp Memory allocation
impl<L: AirParameters> AirBuilder<L> {
    pub fn lookup_table<T: EvalCubic>(
        &mut self,
        challenge: &CubicRegister,
        table: &[T],
    ) -> LookupTable<T, L::Field, L::CubicParams> {
        let multiplicities = self.alloc_array::<ElementRegister>(table.len());
        let multiplicities_table_log = self.alloc_array_extended::<CubicRegister>(table.len());
        let table_accumulator = self.alloc_extended::<CubicRegister>();
        LookupTable {
            challenge: *challenge,
            table: table.to_vec(),
            multiplicities,
            multiplicities_table_log,
            table_accumulator,
            digest: table_accumulator,
            _marker: PhantomData,
        }
    }

    pub fn lookup_table_with_multiplicities<T: EvalCubic>(
        &mut self,
        challenge: &CubicRegister,
        table: &[T],
        multiplicities: &ArrayRegister<ElementRegister>,
    ) -> LookupTable<T, L::Field, L::CubicParams> {
        let multiplicities_table_log = self.alloc_array_extended::<CubicRegister>(table.len());
        let table_accumulator = self.alloc_extended::<CubicRegister>();
        LookupTable {
            challenge: *challenge,
            table: table.to_vec(),
            multiplicities: *multiplicities,
            multiplicities_table_log,
            table_accumulator,
            digest: table_accumulator,
            _marker: PhantomData,
        }
    }

    pub fn lookup_values<T: EvalCubic>(
        &mut self,
        challenge: &CubicRegister,
        values: &[T],
    ) -> LogLookupValues<T, L::Field, L::CubicParams> {
        let mut trace_values = Vec::new();
        let mut public_values = Vec::new();

        for value in values.iter() {
            match value.register() {
                MemorySlice::Public(..) => public_values.push(*value),
                MemorySlice::Local(..) => trace_values.push(*value),
                MemorySlice::Next(..) => unreachable!("Next register not supported for lookup"),
                MemorySlice::Global(..) => public_values.push(*value),
                MemorySlice::Challenge(..) => unreachable!("Cannot send challenge register"),
            }
        }
        assert_eq!(
            trace_values.len() % 2,
            0,
            "Only even number of values supported"
        );

        assert_eq!(
            public_values.len() % 2,
            0,
            "Only even number of values supported"
        );
        let row_accumulators = self.alloc_array_extended::<CubicRegister>(trace_values.len() / 2);
        let global_accumulators = self.alloc_array_global::<CubicRegister>(public_values.len() / 2);
        let log_lookup_accumulator = self.alloc_extended::<CubicRegister>();

        let (digest, global_digest) = match public_values.len() {
            0 => (log_lookup_accumulator, None),
            _ => (
                self.alloc_global::<CubicRegister>(),
                Some(self.alloc_global::<CubicRegister>()),
            ),
        };

        LogLookupValues {
            challenge: *challenge,
            trace_values: trace_values.to_vec(),
            public_values: public_values.to_vec(),
            row_accumulators,
            global_accumulators,
            log_lookup_accumulator,
            local_digest: log_lookup_accumulator,
            digest,
            global_digest,
            _marker: PhantomData,
        }
    }

    pub fn element_lookup_from_table_and_values(
        &mut self,
        table_data: LookupTable<ElementRegister, L::Field, L::CubicParams>,
        values_data: LogLookupValues<ElementRegister, L::Field, L::CubicParams>,
    ) {
        let table_challenge = table_data.challenge;
        let values_challenge = values_data.challenge;
        assert_eq!(
            table_challenge, values_challenge,
            "Challenges must be equal"
        );
        let lookup_data = Lookup::Element(LogLookup {
            table_data: table_data.clone(),
            values_data: values_data.clone(),
            _marker: core::marker::PhantomData,
        });

        // Add the lookup constraints
        // Digest constraints
        self.constraints.push(Constraint::lookup(
            LookupConstraint::<ElementRegister, _, _>::Digest(
                table_data.digest,
                values_data.digest,
            )
            .into(),
        ));
        // table constraints
        self.constraints.push(Constraint::lookup(
            LookupConstraint::Table(table_data).into(),
        ));
        // Values constraints
        self.constraints
            .push(Constraint::lookup(values_data.digest_constraint().into()));
        self.constraints.push(Constraint::lookup(
            LookupConstraint::<ElementRegister, _, _>::ValuesLocal(values_data.clone()).into(),
        ));
        if values_data.global_digest.is_some() {
            self.global_constraints.push(Constraint::lookup(
                LookupConstraint::<ElementRegister, _, _>::ValuesGlobal(values_data.clone()).into(),
            ));
        }

        // Add the lookup to the list of lookups
        self.lookup_data.push(lookup_data);
    }

    pub fn cubic_lookup_from_table_and_values(
        &mut self,
        table_data: LookupTable<CubicRegister, L::Field, L::CubicParams>,
        values_data: LogLookupValues<CubicRegister, L::Field, L::CubicParams>,
    ) {
        let table_challenge = table_data.challenge;
        let values_challenge = values_data.challenge;
        assert_eq!(
            table_challenge, values_challenge,
            "Challenges must be equal"
        );
        let lookup_data = Lookup::CubicElement(LogLookup {
            table_data: table_data.clone(),
            values_data: values_data.clone(),
            _marker: core::marker::PhantomData,
        });

        // Add the lookup constraints

        // Digest constraints
        self.constraints.push(Constraint::lookup(
            LookupConstraint::<CubicRegister, _, _>::Digest(table_data.digest, values_data.digest)
                .into(),
        ));

        // table constraints
        self.constraints.push(Constraint::lookup(
            LookupConstraint::Table(table_data).into(),
        ));
        // Values constraints
        self.constraints
            .push(Constraint::lookup(values_data.digest_constraint().into()));
        self.constraints.push(Constraint::lookup(
            LookupConstraint::<CubicRegister, _, _>::ValuesLocal(values_data.clone()).into(),
        ));
        if values_data.global_digest.is_some() {
            self.global_constraints.push(Constraint::lookup(
                LookupConstraint::<CubicRegister, _, _>::ValuesGlobal(values_data.clone()).into(),
            ));
        }

        // Add the lookup to the list of lookups
        self.lookup_data.push(lookup_data);
    }

    pub fn element_lookup(
        &mut self,
        table: &[ElementRegister],
        values: &[ElementRegister],
    ) -> ArrayRegister<ElementRegister> {
        // Allocate memory for the lookup
        let challenge = self.alloc_challenge::<CubicRegister>();

        let table_data = self.lookup_table(&challenge, table);
        let values_data = self.lookup_values(&challenge, values);
        let multiplicities = table_data.multiplicities;

        self.element_lookup_from_table_and_values(table_data, values_data);

        multiplicities
    }

    pub fn cubic_lookup(
        &mut self,
        table: &[CubicRegister],
        values: &[CubicRegister],
    ) -> ArrayRegister<ElementRegister> {
        // Allocate memory for the lookup
        let challenge = self.alloc_challenge::<CubicRegister>();

        let table_data = self.lookup_table(&challenge, table);
        let values_data = self.lookup_values(&challenge, values);
        let multiplicities = table_data.multiplicities;

        self.cubic_lookup_from_table_and_values(table_data, values_data);

        multiplicities
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::AirParameters;
    use crate::math::extension::cubic::element::CubicElement;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct LookupTest<const N: usize, const M: usize>;

    impl<const N: usize, const M: usize> AirParameters for LookupTest<N, M> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = M + 2 * N;
        const EXTENDED_COLUMNS: usize = (3 * (M / 2)) + 3 * N + 2 * 3;

        type Instruction = EmptyInstruction<GoldilocksField>;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_lookup() {
        type L = LookupTest<N, M>;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;
        const N: usize = 29;
        const M: usize = 10;

        let mut builder = AirBuilder::<L>::new();

        let table_values = builder
            .alloc_array::<ElementRegister>(N)
            .into_iter()
            .collect::<Vec<_>>();
        let values = builder
            .alloc_array::<ElementRegister>(M)
            .into_iter()
            .collect::<Vec<_>>();

        let multiplicities = builder.element_lookup(&table_values, &values);

        let (air, trace_data) = builder.build();

        let num_rows = 1<<16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        // Set the table vals
        for i in 0..num_rows {
            let table_vals = [GoldilocksField::rand(); N];
            for (reg, val) in table_values.iter().zip(table_vals) {
                writer.write(reg, &val, i);
            }
        }

        let mut rng = thread_rng();
        // Se the lookup vals
        for i in 0..num_rows {
            let j_vals = [rng.gen_range(0..num_rows); M];
            let k_vals = [rng.gen_range(0..N); M];
            for (value, (&j, &k)) in values.iter().zip(j_vals.iter().zip(k_vals.iter())) {
                let val = writer.read(&table_values[k], j);
                let mult_value = writer.read(&multiplicities.get(k), j);
                writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
                writer.write(value, &val, i);
            }
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct CubicLookupTest<const N: usize, const M: usize>;

    impl<const N: usize, const M: usize> AirParameters for CubicLookupTest<N, M> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 3 * M + 2 * 3 * N;
        const EXTENDED_COLUMNS: usize = (3 * (3 * M / 2)) + 3 * 3 * N + 2 * 3;

        type Instruction = EmptyInstruction<GoldilocksField>;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_cubic_lookup() {
        type L = CubicLookupTest<N, M>;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;
        const N: usize = 4;
        const M: usize = 102;

        let mut builder = AirBuilder::<L>::new();

        let table_values = builder
            .alloc_array::<CubicRegister>(N)
            .into_iter()
            .collect::<Vec<_>>();
        let values = builder
            .alloc_array::<CubicRegister>(M)
            .into_iter()
            .collect::<Vec<_>>();

        let multiplicities = builder.cubic_lookup(&table_values, &values);

        let (air, trace_data) = builder.build();

        let num_rows = 1<<16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        // Set the table vals
        for i in 0..num_rows {
            let table_vals = [CubicElement::from_slice(&[GoldilocksField::rand(); 3]); N];
            for (reg, val) in table_values.iter().zip(table_vals) {
                writer.write(reg, &val, i);
            }
        }

        let mut rng = thread_rng();
        // Se the lookup vals
        for i in 0..num_rows {
            let j_vals = [rng.gen_range(0..num_rows); M];
            let k_vals = [rng.gen_range(0..N); M];
            for (value, (&j, &k)) in values.iter().zip(j_vals.iter().zip(k_vals.iter())) {
                let val = writer.read(&table_values[k], j);
                let mult_value = writer.read(&multiplicities.get(k), j);
                writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
                writer.write(value, &val, i);
            }
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }

    #[test]
    fn test_cubic_lookup_with_public_values() {
        type L = CubicLookupTest<N, M>;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;
        const N: usize = 4;
        const M: usize = 102;
        const PUB: usize = 10;
        const GLOB: usize = 2;

        let mut builder = AirBuilder::<L>::new();

        let table_values = builder
            .alloc_array::<CubicRegister>(N)
            .into_iter()
            .collect::<Vec<_>>();
        let trace_values = builder
            .alloc_array::<CubicRegister>(M)
            .into_iter()
            .collect::<Vec<_>>();

        let public_values = builder
            .alloc_array_public::<CubicRegister>(PUB)
            .into_iter()
            .collect::<Vec<_>>();

        let global_values = builder
            .alloc_array_global::<CubicRegister>(GLOB)
            .into_iter()
            .collect::<Vec<_>>();

        let values = trace_values
            .iter()
            .copied()
            .chain(public_values.iter().copied())
            .chain(global_values.iter().copied())
            .collect::<Vec<_>>();

        // let multiplicities = builder.lookup_cubic_log_derivative(&table_values, &values);
        let challenge = builder.alloc_challenge::<CubicRegister>();
        let lookup_table = builder.lookup_table(&challenge, &table_values);
        let lookup_values = builder.lookup_values(&challenge, &values);
        assert_eq!(lookup_values.trace_values.len(), M);
        assert_eq!(lookup_values.public_values.len(), PUB + GLOB);
        let multiplicities = lookup_table.multiplicities;
        builder.cubic_lookup_from_table_and_values(lookup_table, lookup_values);

        let (air, trace_data) = builder.build();

        let num_rows = 1<<16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        // Set the table vals
        for i in 0..num_rows {
            let table_vals = [CubicElement::from_slice(&[GoldilocksField::rand(); 3]); N];
            for (reg, val) in table_values.iter().zip(table_vals) {
                writer.write(reg, &val, i);
            }
        }

        let mut rng = thread_rng();
        // Set the lookup vals
        for i in 0..num_rows {
            let j_vals = [rng.gen_range(0..num_rows); M];
            let k_vals = [rng.gen_range(0..N); M];
            for (value, (&j, &k)) in values.iter().zip(j_vals.iter().zip(k_vals.iter())) {
                let val = writer.read(&table_values[k], j);
                let mult_value = writer.read(&multiplicities.get(k), j);
                writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
                writer.write(value, &val, i);
            }
        }

        // Set the public values
        let mut public_inputs: Vec<F> = vec![F::ZERO; air.num_public_values];
        for public_value in public_values.iter() {
            let j = rng.gen_range(0..num_rows);
            let k = rng.gen_range(0..N);
            let val = writer.read(&table_values[k], j);
            public_value.assign_to_raw_slice(&mut public_inputs, &val);
            let mult_value = writer.read(&multiplicities.get(k), j);
            writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
        }

        for global_value in global_values.iter() {
            let j = rng.gen_range(0..num_rows);
            let k = rng.gen_range(0..N);
            let val = writer.read(&table_values[k], j);
            writer.write(global_value, &val, 0);
            let mult_value = writer.read(&multiplicities.get(k), j);
            writer.write(&multiplicities.get(k), &(mult_value + F::ONE), j);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
