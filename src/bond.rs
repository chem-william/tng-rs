#[derive(Debug)]
pub struct Bond {
    /// One of the atoms of the bond
    pub(crate) from_atom_id: i64,
    /// The other atom of the bond
    pub(crate) to_atom_id: i64,
}

impl Bond {
    pub(crate) fn new() -> Self {
        Self {
            from_atom_id: 0,
            to_atom_id: 0,
        }
    }
}
