pub mod builder_operations;
pub mod multiplicity_data;
pub mod table;

use self::multiplicity_data::MultiplicityData;
use super::operations::{ByteOperation, NUM_BIT_OPPS};
use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::table::lookup::log_der::{LogLookupValues, LookupTable};
use crate::chip::AirParameters;

pub const NUM_CHALLENGES: usize = 1 + 6 * 3;
