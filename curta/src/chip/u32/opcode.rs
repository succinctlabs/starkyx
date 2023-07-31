use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct U32Opcode {
    ident: u32,
    pub id: ElementRegister,
    pub a: ElementRegister,
    pub b: ElementRegister,
    pub result: ElementRegister,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn u32_opcode(&mut self, ident: u32, a: ElementRegister, b: ElementRegister) -> U32Opcode {
        let result = self.alloc::<ElementRegister>();
        let id = self.alloc::<ElementRegister>();
        let id_val = L::Field::from_canonical_u32(ident);
        self.assert_expressions_equal(id.expr(), id_val.into());
        U32Opcode {
            ident,
            id,
            a,
            b,
            result,
        }
    }
}

impl<F: PrimeField64> Instruction<F> for U32Opcode {
    fn inputs(&self) -> Vec<MemorySlice> {
        vec![*self.a.register(), *self.b.register()]
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![*self.id.register(), *self.result.register()]
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let id_val = F::from_canonical_u32(self.ident);
        writer.write(&self.id, &id_val, row_index);

        match self.ident {
            100 => {
                let a = writer.read(&self.a, row_index).as_canonical_u64() as u32;
                let b = writer.read(&self.b, row_index).as_canonical_u64() as u32;
                let result = a & b;
                writer.write(&self.result, &F::from_canonical_u32(result), row_index);
            }
            _ => panic!("Invalid opcode"),
        }
    }
}
