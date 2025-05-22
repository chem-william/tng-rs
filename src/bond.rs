#[derive(Debug, Clone, Copy)]
pub struct Bond {
    /// One of the atoms of the bond
    pub from_atom_id: i64,
    /// The other atom of the bond
    pub to_atom_id: i64,
}

impl Bond {
    pub fn new() -> Self {
        Self {
            from_atom_id: 0,
            to_atom_id: 0,
        }
    }
}
