use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;

pub trait Config: 'static + Send + Sync + Sized {
    /// The fieild of the stark.
    type Field: Field;
    /// The packed version of the field used in evaluation.
    type Packing: PackedField<Scalar = Self::Field>;
}
