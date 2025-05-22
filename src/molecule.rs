use crate::{atom::Atom, bond::Bond, chain::Chain, residue::Residue};

#[derive(Debug, Default)]
pub struct Molecule {
    /// A unique ID number of the molecule
    id: i64,
    /// Quaternary structure of the molecule
    /// 1 => monomeric, 2 => dimeric, etc.
    quaternary_str: i64,
    /// The number of chains in the molecule
    n_chains: i64,
    /// The number of residues in the molecule
    n_residues: i64,
    /// The number of atoms in the molecule
    n_atoms: i64,
    /// The number of bonds in the molecule
    /// If the bonds are not specified this value can be 0
    n_bonds: i64,
    /// The name of the molecule
    name: String,
    /// A list of chains in the molecule
    chains: Vec<Chain>,
    /// A list of residues in the molecule
    residues: Vec<Residue>,
    /// A list of the atoms in the molecule
    atoms: Vec<Atom>,
    /// A list of the bonds in the molecule
    bonds: Vec<Bond>,
}

impl Molecule {
    pub fn new() -> Self {
        Self {
            id: 0,
            quaternary_str: 1,
            name: String::new(),
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
}
