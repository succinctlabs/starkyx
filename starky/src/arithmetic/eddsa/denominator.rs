use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::arithmetic::util::{extract_witness_and_shift, split_digits, to_field_iter};
use crate::arithmetic::{ArithmeticParser, Register};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};


