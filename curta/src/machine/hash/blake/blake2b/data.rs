use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::register::U64Register;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::math::field::Field;

pub struct BLAKE2BData<L: AirParameters> {
    pub public: BLAKE2BPublicData,
    pub trace: BLAKE2BTraceData,
    pub memory: BLAKE2BMemory,
    pub consts: BLAKE2BConsts<L>,
    pub const_nums: BLAKE2BConstNums,
}

pub struct BLAKE2BPublicData {
    pub padded_chunks: Vec<ArrayRegister<U64Register>>,
    pub t_values: ArrayRegister<U64Register>,
    pub end_bits: ArrayRegister<BitRegister>,
    pub digest_indices: ArrayRegister<ElementRegister>,
}

pub struct BLAKE2BTraceData {
    pub(crate) clk: ElementRegister,
    pub(crate) is_hash_initialize: BitRegister,
    pub(crate) is_compress_initialize: BitRegister,
    pub(crate) is_compress_first_row: BitRegister,
    pub(crate) is_compress_third_row: BitRegister,
    pub(crate) at_first_compress: BitRegister,
    pub(crate) at_last_hash_compress: BitRegister,
    pub(crate) cycle_96_end_bit: BitRegister,
    pub(crate) digest_bit: Slice<BitRegister>,
    pub(crate) save_final_v: Slice<BitRegister>,
    pub(crate) compress_id: ElementRegister,
    pub(crate) compress_index: ElementRegister,
    pub(crate) compress_iteration: ElementRegister,
    pub(crate) mix_index: ElementRegister,
}

pub struct BLAKE2BMemory {
    pub(crate) h: Slice<U64Register>,
    pub(crate) v: Slice<U64Register>,
    pub(crate) v_final: Slice<U64Register>,
    pub(crate) m: Slice<U64Register>,
    pub(crate) t: Slice<U64Register>,
}

pub struct BLAKE2BConsts<L: AirParameters> {
    pub(crate) iv_values: ArrayRegister<U64Register>,
    pub(crate) compress_iv: Slice<U64Register>,
    pub(crate) v_indices: MemoryArray<L, 8, 4>,
    pub(crate) v_last_write_ages: MemoryArray<L, 8, 4>,
    pub(crate) permutations: MemoryArray<L, 12, 16>,
    pub(crate) dummy_index: ElementRegister,
    pub(crate) dummy_ts: ElementRegister,
    pub(crate) first_compress_h_read_ts: ElementRegister,
}

pub struct BLAKE2BConstNums {
    pub(crate) const_0: ElementRegister,
    pub(crate) const_0_u64: U64Register,
    pub(crate) const_1: ElementRegister,
    pub(crate) const_2: ElementRegister,
    pub(crate) const_3: ElementRegister,
    pub(crate) const_4: ElementRegister,
    pub(crate) const_16: ElementRegister,
    pub(crate) const_96: ElementRegister,
    pub(crate) const_97: ElementRegister,
    pub(crate) const_184: ElementRegister,
    pub(crate) const_192: ElementRegister,
    pub(crate) const_ffffffffffffffff: U64Register,
}

pub(crate) struct MemoryArray<L: AirParameters, const R: usize, const C: usize> {
    pub flattened_memory: Slice<ElementRegister>,
    c_const: ElementRegister,
    _marker: std::marker::PhantomData<L>,
}

impl<L: AirParameters, const R: usize, const C: usize> MemoryArray<L, R, C> {
    pub(crate) fn new(builder: &mut BytesBuilder<L>) -> Self {
        Self {
            flattened_memory: builder.uninit_slice(),
            c_const: builder.constant(&L::Field::from_canonical_usize(C)),
            _marker: core::marker::PhantomData,
        }
    }

    pub(crate) fn store_row(
        &mut self,
        builder: &mut BytesBuilder<L>,
        row: usize,
        values: &[u8],
        mul: ElementRegister,
    ) {
        assert_eq!(values.len(), C);
        assert!(row < R);

        for (i, value) in values.iter().enumerate() {
            let value_const = builder.constant(&L::Field::from_canonical_u8(*value));
            builder.store(
                &self.flattened_memory.get(row * C + i),
                value_const,
                &Time::zero(),
                Some(mul),
            );
        }
    }

    pub(crate) fn get_at(
        &self,
        builder: &mut BytesBuilder<L>,
        row: ElementRegister,
        col: ElementRegister,
    ) -> ElementRegister {
        let mut idx = builder.mul(row, self.c_const);
        idx = builder.add(idx, col);
        builder.load(&self.flattened_memory.get_at(idx), &Time::zero())
    }
}
