use super::parser::AirParser;

#[derive(Debug, Clone)]
pub struct AirOpeningSet<AP: AirParser> {
    pub local_values: Vec<AP::Var>,
    pub next_values: Vec<AP::Var>,
    pub quotient_values: Vec<AP::Var>,
}
