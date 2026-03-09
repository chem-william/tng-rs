use crate::{atom::Atom, bond::Bond, chain::Chain, residue::Residue};

#[derive(Debug, Default)]
pub struct Molecule {
    /// A unique ID number of the molecule
    pub id: i64,
    /// Quaternary structure of the molecule
    /// 1 => monomeric, 2 => dimeric, etc.
    pub quaternary_str: i64,
    /// The number of chains in the molecule
    pub n_chains: i64,
    /// The number of residues in the molecule
    pub n_residues: i64,
    /// The number of atoms in the molecule
    pub n_atoms: i64,
    /// The number of bonds in the molecule
    /// If the bonds are not specified this value can be 0
    pub n_bonds: i64,
    /// The name of the molecule
    pub name: String,
    /// A list of chains in the molecule
    pub chains: Vec<Chain>,
    /// A list of residues in the molecule
    pub residues: Vec<Residue>,
    /// A list of the atoms in the molecule
    pub atoms: Vec<Atom>,
    /// A list of the bonds in the molecule
    pub bonds: Vec<Bond>,
}

impl Molecule {
    pub fn new() -> Self {
        Self {
            quaternary_str: 1,
            name: String::new(),
            id: 0,
            n_chains: 0,
            chains: Vec::new(),
            n_residues: 0,
            residues: Vec::new(),
            n_atoms: 0,
            atoms: Vec::new(),
            n_bonds: 0,
            bonds: Vec::new(),
        }
    }

    /// Rust-side equivalent of C's `tng_molecule_chain_find`.
    ///
    /// This is exposed on [`Molecule`] rather than [`crate::trajectory::Trajectory`]
    /// because the molecule owns its chains in the Rust model.
    pub fn chain_find(&self, name: &str, id: i64) -> Option<&Chain> {
        let chain_out = self.chains.iter().rev().find(|chain| {
            let name_match = name.is_empty() || chain.name == name;
            let id_match = id == -1 || chain.id as i64 == id;
            name_match && id_match
        });

        chain_out
    }

    /// Rust-side equivalent of C's `tng_molecule_atom_find`.
    ///
    /// This is exposed on [`Molecule`] rather than [`crate::trajectory::Trajectory`]
    /// because the molecule owns its atoms in the Rust model.
    pub fn atom_find(&self, name: &str, id: i64) -> Option<&Atom> {
        let atom_out = self.atoms.iter().rev().find(|atom| {
            let name_match = name.is_empty() || atom.name == name;
            let id_match = id == -1 || atom.id == id;
            name_match && id_match
        });

        atom_out
    }

    /// Rust convenience for residue lookup within a molecule.
    ///
    /// There is no direct C equivalent named `tng_molecule_residue_find`; the
    /// closest C search API is `tng_chain_residue_find`.
    pub fn residue_find(&self, name: &str, id: i64) -> Option<&Residue> {
        let residue_out = self.residues.iter().rev().find(|residue| {
            let name_match = name.is_empty() || residue.name == name;
            let id_match = id == -1 || residue.id as i64 == id;
            name_match && id_match
        });

        residue_out
    }

    /// Retrieve the atom of a residue with specified index in the list of atoms
    ///
    /// # Panic
    /// Panics if `index` + `residue.atoms_offset` is out of bounds for `self.atoms`
    pub fn residue_atom_of_index(&self, index: usize, residue: &Residue) -> Atom {
        self.atoms[residue.atoms_offset + index].clone()
    }
}
