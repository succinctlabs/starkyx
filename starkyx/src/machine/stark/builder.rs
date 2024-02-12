use super::Stark;
use crate::chip::builder::AirBuilder;
use crate::chip::register::element::ElementRegister;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::Starky;

pub struct StarkBuilder<L: AirParameters> {
    pub api: AirBuilder<L>,
    pub clk: ElementRegister,
}

impl<L: AirParameters> Builder for StarkBuilder<L> {
    type Field = L::Field;
    type CubicParams = L::CubicParams;
    type Parameters = L;
    type Instruction = L::Instruction;

    fn api(&mut self) -> &mut AirBuilder<Self::Parameters> {
        &mut self.api
    }

    fn clk(&mut self) -> ElementRegister {
        self.clk
    }
}

impl<L: AirParameters> StarkBuilder<L> {
    pub fn new() -> Self {
        let mut api = AirBuilder::<L>::new();
        let clk = api.clock();
        api.init_local_memory();
        StarkBuilder { api, clk }
    }

    pub fn build<C: CurtaConfig<D, F = L::Field>, const D: usize>(
        self,
        num_rows: usize,
    ) -> Stark<L, C, D> {
        let api = self.api;

        let config = StarkyConfig::<C, D>::standard_fast_config(num_rows);
        let (air, air_data) = api.build();
        let stark = Starky::new(air);

        Stark {
            config,
            stark,
            air_data,
        }
    }
}
