use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::uint::register::U32Register;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    pub fn sha_256_round(&mut self, chunk: ArrayRegister<U32Register>) {
        let cycle = self.cycle(4);

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
