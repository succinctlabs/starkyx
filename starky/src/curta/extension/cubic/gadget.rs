use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::array::CubicArray;

#[derive(Debug, Clone, Copy)]
pub struct CubicGadget<F, const D: usize>(core::marker::PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> CubicGadget<F, D> {
    pub fn const_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: [F::Extension; 3],
    ) -> CubicArray<ExtensionTarget<D>> {
        CubicArray([
            builder.constant_extension(a[0]),
            builder.constant_extension(a[1]),
            builder.constant_extension(a[2]),
        ])
    }

    pub fn zero_extension(builder: &mut CircuitBuilder<F, D>) -> CubicArray<ExtensionTarget<D>> {
        CubicArray([
            builder.zero_extension(),
            builder.zero_extension(),
            builder.zero_extension(),
        ])
    }

    pub fn one_extension(builder: &mut CircuitBuilder<F, D>) -> CubicArray<ExtensionTarget<D>> {
        CubicArray([
            builder.one_extension(),
            builder.zero_extension(),
            builder.zero_extension(),
        ])
    }

    pub fn from_base_extension(
        builder: &mut CircuitBuilder<F, D>,
        element: ExtensionTarget<D>,
    ) -> CubicArray<ExtensionTarget<D>> {
        CubicArray([element, builder.zero_extension(), builder.zero_extension()])
    }

    pub fn add_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: &CubicArray<ExtensionTarget<D>>,
        b: &CubicArray<ExtensionTarget<D>>,
    ) -> CubicArray<ExtensionTarget<D>> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicArray([
            builder.add_extension(x_0, y_0),
            builder.add_extension(x_1, y_1),
            builder.add_extension(x_2, y_2),
        ])
    }

    pub fn sub_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: &CubicArray<ExtensionTarget<D>>,
        b: &CubicArray<ExtensionTarget<D>>,
    ) -> CubicArray<ExtensionTarget<D>> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicArray([
            builder.sub_extension(x_0, y_0),
            builder.sub_extension(x_1, y_1),
            builder.sub_extension(x_2, y_2),
        ])
    }

    pub fn mul_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: &CubicArray<ExtensionTarget<D>>,
        b: &CubicArray<ExtensionTarget<D>>,
    ) -> CubicArray<ExtensionTarget<D>> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);

        let x_0y_0 = builder.mul_extension(x_0, y_0);
        let x_0y_1 = builder.mul_extension(x_0, y_1);
        let x_0y_2 = builder.mul_extension(x_0, y_2);
        let x_1y_0 = builder.mul_extension(x_1, y_0);
        let x_1y_1 = builder.mul_extension(x_1, y_1);
        let x_1y_2 = builder.mul_extension(x_1, y_2);
        let x_2y_0 = builder.mul_extension(x_2, y_0);
        let x_2y_1 = builder.mul_extension(x_2, y_1);
        let x_2y_2 = builder.mul_extension(x_2, y_2);

        let mut z_0 = builder.sub_extension(x_0y_0, x_1y_2);
        z_0 = builder.sub_extension(z_0, x_2y_1);

        let mut z_1 = builder.add_extension(x_0y_1, x_1y_0);
        z_1 = builder.add_extension(z_1, x_1y_2);
        z_1 = builder.add_extension(z_1, x_2y_1);
        z_1 = builder.sub_extension(z_1, x_2y_2);

        let mut z_2 = builder.add_extension(x_0y_2, x_1y_1);
        z_2 = builder.add_extension(z_2, x_2y_0);
        z_2 = builder.add_extension(z_2, x_2y_2);

        CubicArray([z_0, z_1, z_2])
    }
}
