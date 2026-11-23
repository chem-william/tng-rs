#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Bond {
    /// One of the atoms of the bond
    pub(crate) from_atom_id: i64,
    /// The other atom of the bond
    pub(crate) to_atom_id: i64,
}
