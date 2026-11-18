use crate::{atom::Atom, bond::Bond, chain::Chain, residue::Residue};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Molecule {
    /// A unique ID number of the molecule
    pub(crate) id: i64,
    /// Quaternary structure of the molecule
    /// 1 => monomeric, 2 => dimeric, etc.
    pub(crate) quaternary_str: i64,
    /// The name of the molecule
    pub(crate) name: String,
    /// A list of chains in the molecule
    pub(crate) chains: Vec<Chain>,
    /// A list of residues in the molecule
    pub(crate) residues: Vec<Residue>,
    /// A list of the atoms in the molecule
    pub(crate) atoms: Vec<Atom>,
    /// A list of the bonds in the molecule. Can be empty if the bonds are not specified.
    pub(crate) bonds: Vec<Bond>,
}

impl Molecule {
    pub(crate) fn new() -> Self {
        Self {
            quaternary_str: 1,
            name: String::new(),
            id: 0,
            chains: Vec::new(),
            residues: Vec::new(),
            atoms: Vec::new(),
            bonds: Vec::new(),
        }
    }

    /// C API: `tng_molecule_chain_find`.
    ///
    /// This is exposed on [`Molecule`] rather than [`crate::trajectory::Trajectory`]
    /// because the molecule owns its chains in the Rust model.
    pub fn chain_find(&self, name: &str, id: i64) -> Option<&Chain> {
        self.chains.iter().rev().find(|chain| {
            let name_match = name.is_empty() || chain.name == name;
            let id_match = id == -1 || chain.id as i64 == id;
            name_match && id_match
        })
    }

    /// C API: `tng_molecule_atom_find`.
    ///
    /// This is exposed on [`Molecule`] rather than [`crate::trajectory::Trajectory`]
    /// because the molecule owns its atoms in the Rust model.
    pub fn atom_find(&self, name: &str, id: i64) -> Option<&Atom> {
        self.atoms.iter().rev().find(|atom| {
            let name_match = name.is_empty() || atom.name == name;
            let id_match = id == -1 || atom.id == id;
            name_match && id_match
        })
    }

    /// No direct C equivalent. Closest C API: `tng_chain_residue_find`.
    pub fn residue_find(&self, name: &str, id: i64) -> Option<&Residue> {
        self.residues.iter().rev().find(|residue| {
            let name_match = name.is_empty() || residue.name == name;
            let id_match = id == -1 || residue.id as i64 == id;
            name_match && id_match
        })
    }

    /// C API: `tng_residue_atom_of_index_get`.
    ///
    /// # Panic
    /// Panics if `index` + `residue.atoms_offset` is out of bounds for `self.atoms`
    pub fn residue_atom_of_index(&self, index: usize, residue: &Residue) -> &Atom {
        &self.atoms[residue.atoms_offset + index]
    }
}
