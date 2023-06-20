use core::hash::Hash;

use super::prelude::{Algebra, Field};

pub trait Packed<F: Field>: Algebra<F> + 'static + Copy + Eq + Hash + Send + Sync {}
