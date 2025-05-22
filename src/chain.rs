use crate::MAX_STR_LEN;
use crate::molecule::Molecule;
use crate::residue::Residue;

#[derive(Debug, Default)]
pub struct Chain {
    /// The molecule containing this chain
    molecule: Molecule,
    /// A unique (per molecule) ID number of the chain
    id: usize,
    /// The name of the chain
    name: String,
    /// The number of residues in the chain
    n_residues: usize,
    /// A list of residues in the chain
    residues: Vec<Residue>,
}

impl Chain {
    pub fn set_name(&mut self, new_name: String) {
        assert!(!new_name.is_empty(), "new_name must not be empty.");

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
