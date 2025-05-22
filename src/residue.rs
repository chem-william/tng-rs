use crate::{trajectory::Trajectory, utils};

#[derive(Debug, Default, Clone)]
pub struct Residue {
    /// The chain containing this residue
    // chain: Chain,
    pub chain_index: Option<usize>,
    /// A unique (per chain) ID number of the residue
    pub id: u64,
    /// The name of the residue
    pub name: String,
    /// The number of atoms in the residue
    pub n_atoms: u64,
    /// A list of atoms in the residue
    pub atoms_offset: usize,
}

impl Residue {
    pub fn new() -> Self {
        Self {
            chain_index: None,
            id: 0,
            name: String::new(),
            n_atoms: 0,
            atoms_offset: 0,
        }
    }
    /// Read the residue data of a molecules block.
    pub fn read_data(&mut self, trajectory_data: &mut Trajectory) {
        let inp_file = trajectory_data
            .input_file
            .as_mut()
            .expect("init input_file");
        self.id = utils::read_u64_le_bytes(inp_file);
        self.name = utils::fread_str(inp_file);
        self.n_atoms = utils::read_u64_le_bytes(inp_file);
    }
}
