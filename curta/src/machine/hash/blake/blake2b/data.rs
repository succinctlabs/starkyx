use super::{MIX_LENGTH, MSG_ARRAY_SIZE, NUM_MIX_ROUNDS};
use crate::chip::memory::instruction::MemorySliceIndex;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::register::U64Register;
use crate::machine::builder::Builder;
use crate::math::field::Field;

pub struct BLAKE2BData<B: Builder> {
    pub public: BLAKE2BPublicData,
    pub trace: BLAKE2BTraceData,
    pub memory: BLAKE2BMemory,
    pub consts: BLAKE2BConsts<B>,
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
    pub(crate) is_compress_initialize: BitRegister,
    pub(crate) is_compress_first_row: BitRegister,
    pub(crate) is_compress_third_row: BitRegister,
    pub(crate) is_compress_final_row: BitRegister,
    pub(crate) is_compress_finalize: BitRegister,
    pub(crate) is_digest_row: BitRegister,
    pub(crate) at_first_compress: BitRegister,
    pub(crate) at_digest_compress: BitRegister,
    pub(crate) at_end_compress: BitRegister,
    pub(crate) at_dummy_compress: BitRegister,
    pub(crate) compress_id: ElementRegister,
    pub(crate) previous_compress_id: ElementRegister,
    pub(crate) compress_index: ElementRegister,
    pub(crate) mix_id: ElementRegister,
    pub(crate) mix_index: ElementRegister,
}

pub struct BLAKE2BMemory {
    pub(crate) h: Slice<U64Register>,
    pub(crate) v: Slice<U64Register>,
    pub(crate) v_final: Slice<U64Register>,
    pub(crate) m: Slice<U64Register>,
    pub(crate) t: Slice<U64Register>,
}

pub struct BLAKE2BConsts<B: Builder> {
    pub(crate) iv: Slice<U64Register>,
    pub(crate) iv_values: ArrayRegister<U64Register>,
    pub(crate) compress_iv: Slice<U64Register>,
    pub(crate) v_indices: MemoryArray<B, MIX_LENGTH, 4>,
    pub(crate) v_last_write_ages: MemoryArray<B, MIX_LENGTH, 4>,
    pub(crate) permutations: MemoryArray<B, NUM_MIX_ROUNDS, MSG_ARRAY_SIZE>,
    pub(crate) dummy_index: ElementRegister,
    pub(crate) dummy_index_2: ElementRegister,
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
    pub(crate) const_8: ElementRegister,
    pub(crate) const_10: ElementRegister,
    pub(crate) const_12: ElementRegister,
    pub(crate) const_16: ElementRegister,
    pub(crate) const_91: ElementRegister,
    pub(crate) const_96: ElementRegister,
    pub(crate) const_ffffffffffffffff: U64Register,
}

pub(crate) struct MemoryArray<B: Builder, const R: usize, const C: usize> {
    pub flattened_memory: Slice<ElementRegister>,
    c_const: ElementRegister,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Builder, const R: usize, const C: usize> MemoryArray<B, R, C> {
    pub(crate) fn new(builder: &mut B) -> Self {
        Self {
            flattened_memory: builder.uninit_slice(),
            c_const: builder.constant(&B::Field::from_canonical_usize(C)),
            _marker: core::marker::PhantomData,
        }
    }

    pub(crate) fn store_row(
        &mut self,
        builder: &mut B,
        row: usize,
        values: &[u8],
        mul: ElementRegister,
        label: Option<String>,
    ) {
        assert_eq!(values.len(), C);
        assert!(row < R);

        for (i, value) in values.iter().enumerate() {
            let value_const = builder.constant(&B::Field::from_canonical_u8(*value));
            builder.store::<ElementRegister>(
                &self.flattened_memory.get(row * C + i),
                value_const,
                &Time::zero(),
                Some(mul),
                label.clone(),
                Some(MemorySliceIndex::Index(row * C + i)),
            );
        }
    }

    pub(crate) fn get_at(
        &self,
        builder: &mut B,
        row: ElementRegister,
        col: ElementRegister,
        label: Option<String>,
    ) -> ElementRegister {
        let mut idx = builder.mul(row, self.c_const);
        idx = builder.add(idx, col);

        builder.load(
            &self.flattened_memory.get_at(idx),
            &Time::zero(),
            label.clone(),
            Some(MemorySliceIndex::IndexElement(idx)),
        )
    }
}
