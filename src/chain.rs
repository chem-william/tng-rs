use crate::trajectory::Trajectory;
use crate::{MAX_STR_LEN, utils};

#[derive(Debug, Default, Clone)]
pub struct Chain {
    // /// The molecule containing this chain
    // pub molecule: Molecule,
    // Instead of the full molecule, we store the molecule id
    // otherwise, we'd have a circular reference
    pub parent_molecule_idx: usize,
    /// A unique (per molecule) ID number of the chain
    pub id: u64,
    /// The name of the chain
    pub name: String,
    /// The number of residues in the chain
    pub n_residues: u64,
    /// A list of residues in the chain
    // pub residues: Vec<Residue>,
    pub residues_indices: (usize, usize),
}

impl Chain {
    pub fn new() -> Self {
        Self {
            parent_molecule_idx: 0,
            id: 0,
            name: String::new(),
            n_residues: 0,
            residues_indices: (0, 0),
        }
    }

    // c function: tng_chain_data_read
    pub fn read_data(&mut self, trajectory_data: &mut Trajectory) {
        let inp_file = trajectory_data
            .input_file
            .as_mut()
            .expect("init input_file");
        self.id = utils::read_u64(
            inp_file,
            trajectory_data.endianness64,
            trajectory_data.input_swap64,
        );
        self.name = utils::fread_str(inp_file);
        self.n_residues = utils::read_u64(
            inp_file,
            trajectory_data.endianness64,
            trajectory_data.input_swap64,
        );
    }

    pub fn set_name(&mut self, new_name: String) {
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
