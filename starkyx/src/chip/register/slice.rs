use core::borrow::Borrow;
use core::fmt::Debug;

use super::array::{ArrayIterator, ArrayRegister};
use super::Register;

/// A trait that enables the interface of a slice uniform on both `ArrayRegister<T>` such as a
/// vector `Vec<T>`, array, or slice `&[T]`.
pub trait RegisterSlice<T: Register>: Debug + Clone + Send + Sync {
    type Item<'a>: Borrow<T>
    where
        Self: 'a;
    type Iterator<'a>: Iterator<Item = Self::Item<'a>>
    where
        Self: 'a;

    fn get_value(&self, index: usize) -> T;

    fn len(&self) -> usize;

    fn first_value(&self) -> Option<T>;

    fn last_value(&self) -> Option<T>;

    fn value_iter(&self) -> Self::Iterator<'_>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Register> RegisterSlice<T> for ArrayRegister<T> {
    type Item<'a> = T;
    type Iterator<'a> = ArrayIterator<T>;

    fn get_value(&self, index: usize) -> T {
        self.get(index)
    }

    fn first_value(&self) -> Option<T> {
        self.first()
    }

    fn last_value(&self) -> Option<T> {
        self.last()
    }

    fn value_iter(&self) -> Self::Iterator<'_> {
        self.iter()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: Register> RegisterSlice<T> for Vec<T> {
    type Item<'a> = &'a T;
    type Iterator<'a> = core::slice::Iter<'a, T>;

    fn get_value(&self, index: usize) -> T {
        self[index]
    }

    fn first_value(&self) -> Option<T> {
        self.first().copied()
    }

    fn last_value(&self) -> Option<T> {
        self.last().copied()
    }

    fn value_iter(&self) -> Self::Iterator<'_> {
        self.iter()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<'b, T: Register> RegisterSlice<T> for &'b [T] {
    type Item<'a> = &'a T where Self: 'a;
    type Iterator<'a> = core::slice::Iter<'a, T> where Self: 'a;

    fn get_value(&self, index: usize) -> T {
        self[index]
    }

    fn first_value(&self) -> Option<T> {
        self.first().copied()
    }

    fn last_value(&self) -> Option<T> {
        self.last().copied()
    }

    fn value_iter(&self) -> Self::Iterator<'_> {
        self.iter()
    }

    fn len(&self) -> usize {
        (*self).len()
    }
}
