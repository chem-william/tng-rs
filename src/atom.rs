use crate::{trajectory::Trajectory, utils};

#[derive(Debug, Clone)]
pub struct Atom {
    // /// The residue containing this atom
    // residue: Residue,
    pub residue_index: Option<u64>,
    /// A unique (per molecule) ID number of the atom
    pub id: i64,
    /// The atom_type (depending on the forcefield)
    pub atom_type: String,
    /// The name of the atom
    pub name: String,
}

impl Atom {
    pub fn new() -> Self {
        Self {
            residue_index: None,
            id: 0,
            name: String::new(),
            atom_type: String::new(),
        }
    }

    // c function name: tng_atom_data_read
    /// Read the atom data of a molecules block
    pub fn read_data(&mut self, trajectory_data: &mut Trajectory) {
        let inp_file = trajectory_data
            .input_file
            .as_mut()
            .expect("init input_file");
        self.id = utils::read_i64_le_bytes(inp_file);
        self.name = utils::fread_str(inp_file);
        self.atom_type = utils::fread_str(inp_file);
    }
}
