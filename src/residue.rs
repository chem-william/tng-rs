use crate::chain::Chain;

#[derive(Debug, Default)]
pub struct Residue {
    /// The chain containing this residue
    chain: Chain,
    /// A unique (per chain) ID number of the residue
    id: usize,
    /// The name of the residue
    name: String,
    /// The number of atoms in the residue
    n_atoms: usize,
    /// A list of atoms in the residue
    atoms_offset: usize,
}
