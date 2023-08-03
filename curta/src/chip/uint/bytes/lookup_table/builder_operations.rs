use super::multiplicity_data::MultiplicityData;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    row_acc_challenges: ArrayRegister<CubicRegister>,
    multiplicities: ArrayRegister<ElementRegister>,
    multiplicity_data: MultiplicityData,
    values: Vec<CubicRegister>,
}
