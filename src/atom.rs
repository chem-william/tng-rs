use crate::residue::Residue;

#[derive(Debug)]
pub struct Atom {
    /// The residue containing this atom
    residue: Residue,
    /// A unique (per molecule) ID number of the atom
    id: i64,
    /// The atom_type (depending on the forcefield)
    atom_type: String,
    /// The name of the atom
    name: String,
}

impl Atom {
    pub fn new() -> Self {
        Self {
            residue: Residue::default(),
            id: 0,
            name: String::new(),
            atom_type: String::new(),
        }
    }
}
