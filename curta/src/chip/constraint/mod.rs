pub enum Constraint<T> {
    First(T),
    Last(T),
    Transition(T),
    All(T),
}
