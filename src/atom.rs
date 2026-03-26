use crate::{trajectory::Trajectory, utils};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom {
    /// The molecule containing this atom.
    pub(crate) parent_molecule_idx: usize,
    // /// The residue containing this atom
    // residue: Residue,
    /// The index of the residue containing this atom
    pub(crate) residue_index: Option<usize>,
    /// A unique (per molecule) ID number of the atom
    pub(crate) id: i64,
    /// The atom_type (depending on the forcefield)
    pub(crate) atom_type: String,
    /// The name of the atom
    pub(crate) name: String,
}

impl Atom {
    pub(crate) fn new() -> Self {
        Self {
            parent_molecule_idx: 0,
            residue_index: None,
            id: 0,
            name: String::new(),
            atom_type: String::new(),
        }
    }

    // c function name: tng_atom_data_read
    /// Read the atom data of a molecules block
    pub(crate) fn read_data(&mut self, trajectory_data: &mut Trajectory) {
        let inp_file = trajectory_data
            .input_file
            .as_mut()
            .expect("init input_file");
        self.id = utils::read_i64(
            inp_file,
            trajectory_data.endianness64,
            trajectory_data.input_swap64,
        );
        self.name = utils::fread_str(inp_file);
        self.atom_type = utils::fread_str(inp_file);
    }
}
