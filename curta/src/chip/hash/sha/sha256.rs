use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::uint::register::U32Register;
use crate::chip::AirParameters;

#[allow(dead_code)]
#[allow(unused_variables)]
impl<L: AirParameters> AirBuilder<L> {
    pub fn sha_256_round(&mut self, chunk: ArrayRegister<U32Register>) {
        let cycle_64 = self.cycle(6);
        let cycle_16 = self.cycle(4);

        // Absorbe values into the sponge
        let w_value = self.alloc::<U32Register>();
        let ab_s_0 = self.alloc::<U32Register>();
        let ab_s_1 = self.alloc::<U32Register>();

        // Put the w value into the bus
        //TODO

        // Sponge compression
        let a = self.alloc::<U32Register>();
        let b = self.alloc::<U32Register>();
        let c = self.alloc::<U32Register>();
        let d = self.alloc::<U32Register>();
        let e = self.alloc::<U32Register>();
        let f = self.alloc::<U32Register>();
        let g = self.alloc::<U32Register>();
        let h = self.alloc::<U32Register>();
    }
}
