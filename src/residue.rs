use std::ops::Range;

use md5::Md5;

use crate::{trajectory::Trajectory, utils};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Residue {
    /// The molecule containing this residue.
    pub(crate) parent_molecule_idx: usize,
    /// The chain containing this residue
    // chain: Chain,
    pub(crate) chain_index: Option<usize>,
    /// A unique (per chain) ID number of the residue
    pub(crate) id: u64,
    /// The name of the residue
    pub(crate) name: String,
    /// The atoms in the residue, as indices into the parent molecule's `atoms`
    pub(crate) atoms: Range<usize>,
}

impl Residue {
    pub(crate) fn new() -> Self {
        Self {
            parent_molecule_idx: 0,
            chain_index: None,
            id: 0,
            name: String::new(),
            atoms: 0..0,
        }
    }
    /// Read the residue data of a molecules block. Returns the number of atoms in the residue,
    /// which the caller uses to set [`Self::atoms`].
    pub(crate) fn read_data(
        &mut self,
        trajectory_data: &mut Trajectory,
        mut hasher: Option<&mut Md5>,
    ) -> u64 {
        let inp_file = trajectory_data
            .input_file
            .as_mut()
            .expect("init input_file");
        self.id = utils::read_u64(
            inp_file,
            trajectory_data.endianness64,
            trajectory_data.input_swap64,
            hasher.as_deref_mut(),
        );
        self.name = utils::fread_str(inp_file, hasher.as_deref_mut());
        utils::read_u64(
            inp_file,
            trajectory_data.endianness64,
            trajectory_data.input_swap64,
            hasher,
        )
    }
}
