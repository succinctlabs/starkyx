// use plonky2::hash::hash_types::RichField;

// use crate::trace::AirTrace;

// use super::stark::config::StarkyConfig;

// impl<F: RichField> AirTrace<F> {
//     fn commit<C, const D: usize>(&self, config: &StarkyConfig<C, D>) {
//         let trace_cols = self.as_columns()
//         .into_par_iter()
//         .map(PolynomialValues::from)
//         .collect::<Vec<_>>();
//     }
// }
