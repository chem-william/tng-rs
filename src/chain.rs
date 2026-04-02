use md5::Md5;

use crate::trajectory::Trajectory;
use crate::{MAX_STR_LEN, utils};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Chain {
    // /// The molecule containing this chain
    // pub molecule: Molecule,
    // Instead of the full molecule, we store the molecule id
    // otherwise, we'd have a circular reference
    pub(crate) parent_molecule_idx: usize,
    /// A unique (per molecule) ID number of the chain
    pub(crate) id: u64,
    /// The name of the chain
    pub(crate) name: String,
    /// The number of residues in the chain
    pub(crate) n_residues: u64,
    /// A list of residues in the chain
    // pub residues: Vec<Residue>,
    pub(crate) residues_indices: (usize, usize),
}

impl Chain {
    pub(crate) fn new() -> Self {
        Self {
            parent_molecule_idx: 0,
            id: 0,
            name: String::new(),
            n_residues: 0,
            residues_indices: (0, 0),
        }
    }

    /// C API: `tng_chain_data_read`
    ///
    /// Read the chain data of a molecules block
    pub(crate) fn read_data(
        &mut self,
        trajectory_data: &mut Trajectory,
        mut hasher: Option<&mut Md5>,
    ) {
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
        self.n_residues = utils::read_u64(
            inp_file,
            trajectory_data.endianness64,
            trajectory_data.input_swap64,
            hasher.as_deref_mut(),
        );
    }

    /// C API: `tng_chain_name_set`
    ///
    /// Set the name of a chain.
    pub(crate) fn set_name(&mut self, new_name: &str) {
        // The C version leaves space for a '\0' in a buffer of size TNG_MAX_STR_LEN.
        // In Rust, Strings don't need a trailing zero, so we just clamp to:
        let max_bytes = MAX_STR_LEN - 1;

        // Truncate at the byte level (safe for ASCII).
        let truncated = if new_name.len() <= max_bytes {
            new_name.to_string()
        } else {
            // Take exactly `max_bytes` bytes from `new_name`.
            // Since it's ASCII, slicing by byte index is valid.
            new_name[..max_bytes].to_string()
        };

        self.name = truncated;
    }
}
