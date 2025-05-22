use crate::{atom::Atom, bond::Bond, chain::Chain, residue::Residue};

#[derive(Debug, Default, Clone)]
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
