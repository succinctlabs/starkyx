use crate::air::parser::AirParser;

pub trait Challenger<AP: AirParser>: Sized {
    type Element: Copy + 'static;

    /// Absorbe the element information into the internal state of the challenger
    fn observe_element(&mut self, parser: &mut AP, element: &Self::Element);

    fn observe_elements<'a, I: IntoIterator<Item = &'a Self::Element>>(
        &mut self,
        parser: &mut AP,
        elements: I,
    ) {
        for element in elements {
            self.observe_element(parser, element);
        }
    }

    /// Get a challenge
    fn challenge(&mut self, parser: &mut AP) -> AP::Var;

    fn challenges_array<const N: usize>(&mut self, parser: &mut AP) -> [AP::Var; N] {
        core::array::from_fn(|_| self.challenge(parser))
    }

    fn challenges_iter<'a>(&'a mut self, parser: &'a mut AP) -> ChallengesIter<'a, Self, AP> {
        ChallengesIter {
            challenger: self,
            state: parser,
            get_challenge: Self::challenge,
        }
    }

    fn challenges_vec(&mut self, parser: &mut AP, n: usize) -> Vec<AP::Var> {
        self.challenges_iter(parser).take(n).collect()
    }
}

pub struct ChallengesIter<'a, C, AP: AirParser> {
    challenger: &'a mut C,
    state: &'a mut AP,
    get_challenge: fn(&mut C, &mut AP) -> AP::Var,
}

impl<C, AP: AirParser> Iterator for ChallengesIter<'_, C, AP> {
    type Item = AP::Var;

    fn next(&mut self) -> Option<Self::Item> {
        Some((self.get_challenge)(self.challenger, self.state))
    }
}
