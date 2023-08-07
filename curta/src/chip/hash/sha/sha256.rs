use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::uint::register::U32Register;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    pub fn sha_256_round(&mut self, chunk: ArrayRegister<U32Register>) {
        let cycle_64 = self.cycle(6);
        let cycle_16 = self.cycle(4);

        // Absorbe values into the sponge

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
