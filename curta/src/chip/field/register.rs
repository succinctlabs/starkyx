use core::iter::once;
use core::marker::PhantomData;

use log::debug;
use serde::{Deserialize, Serialize};

use super::parameters::FieldParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::memory::pointer::raw::RawPointer;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cell::CellType;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::u16::U16Register;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::math::prelude::*;
use crate::polynomial::Polynomial;

/// A register for representing a field element. The value is decomposed into a series of U16 limbs
/// which is controlled by `NB_LIMBS` in FieldParameters. Each limb is range checked using a lookup.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FieldRegister<P: FieldParameters> {
    register: MemorySlice,
    _marker: PhantomData<P>,
}

impl<P: FieldParameters> RegisterSerializable for FieldRegister<P> {
    const CELL: CellType = CellType::U16;

    fn register(&self) -> &MemorySlice {
        &self.register
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self {
            register,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<P: FieldParameters> RegisterSized for FieldRegister<P> {
    fn size_of() -> usize {
        P::NB_LIMBS
    }
}

impl<P: FieldParameters> Register for FieldRegister<P> {
    type Value<T> = Polynomial<T>;

    fn value_from_slice<T: Clone>(slice: &[T]) -> Self::Value<T> {
        Polynomial::from_coefficients_slice(slice)
    }

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        &value.coefficients
    }
}

impl<P: FieldParameters> MemoryValue for FieldRegister<P> {
    fn num_challenges() -> usize {
        assert_eq!(P::NB_LIMBS % 2, 0);
        P::NB_LIMBS / 2 + 1
    }

    fn compress<L: crate::chip::AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        ptr: RawPointer,
        time: &Time<L::Field>,
        challenges: &ArrayRegister<CubicRegister>,
    ) -> CubicRegister {
        let limb_array = ArrayRegister::<U16Register>::from_register_unsafe(self.register);
        assert_eq!(limb_array.len(), P::NB_LIMBS);
        let expressions = (0..P::NB_LIMBS)
            .step_by(2)
            .map(|i| {
                let pair = limb_array.get_subarray(i..i + 2);
                pair.get(0).expr() + pair.get(1).expr() * L::Field::from_canonical_u32(1 << 16)
            })
            .chain(once(time.expr()))
            .collect::<Vec<_>>();
        let compressed = if self.is_trace() {
            builder.accumulate_expressions(challenges, &expressions)
        } else {
            builder.accumulate_public_expressions(challenges, &expressions)
        };

        ptr.accumulate_cubic(builder, compressed.ext_expr())
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use num::BigUint;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use rand::thread_rng;
    use serde::{Deserialize, Serialize};

    use crate::chip::builder::tests::ArithmeticGenerator;
    use crate::chip::builder::AirBuilder;
    use crate::chip::field::instruction::FpInstruction;
    use crate::chip::field::parameters::tests::Fp25519;
    use crate::chip::field::parameters::FieldParameters;
    use crate::chip::field::register::FieldRegister;
    use crate::chip::memory::time::Time;
    use crate::chip::AirParameters;
    use crate::machine::builder::Builder;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::math::prelude::*;
    use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    use crate::plonky2::stark::tests::{test_recursive_starky, test_starky};
    use crate::plonky2::stark::Starky;
    use crate::polynomial::Polynomial;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    struct FpMemAddTest;

    impl AirParameters for FpMemAddTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 213;

        type Instruction = FpInstruction<Fp25519>;
    }

    #[test]
    fn test_fp_memory_add() {
        type F = GoldilocksField;
        type L = FpMemAddTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type P = Fp25519;

        let p = Fp25519::modulus();

        let _ = env_logger::builder().is_test(true).try_init();

        let num_rows = 1 << 16;

        let mut builder = AirBuilder::<L>::new();
        builder.init_local_memory();

        let a_pub = builder.alloc_public::<FieldRegister<P>>();
        let b_pub = builder.alloc_public::<FieldRegister<P>>();

        let a_ptr = builder.uninit::<FieldRegister<P>>();
        let b_ptr = builder.uninit::<FieldRegister<P>>();

        let mult = builder.constant(&F::from_canonical_usize(num_rows));

        builder.store(&a_ptr, a_pub, &Time::zero(), Some(mult));
        builder.store(&b_ptr, b_pub, &Time::zero(), Some(mult));

        let a = builder.load(&a_ptr, &Time::zero());
        let b = builder.load(&b_ptr, &Time::zero());
        let _ = builder.add(a, b);

        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let writer = generator.new_writer();

        let mut rng = thread_rng();

        let a_int: BigUint = rng.gen_biguint(256) % &p;
        let b_int = rng.gen_biguint(256) % &p;
        let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
        let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);
        writer.write(&a_pub, &p_a, 0);
        writer.write(&b_pub, &p_b, 0);
        writer.write_global_instructions(&generator.air_data);
        for i in 0..num_rows {
            writer.write_row_instructions(&generator.air_data, i);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        let writer = generator.new_writer();
        let public = writer.public().unwrap().clone();
        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }
}
