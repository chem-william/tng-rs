use flate2::Decompress;
use log::warn;

use crate::atom::Atom;
use crate::bond::Bond;
use crate::chain::Chain;
use crate::data::{Compression, Data};
use crate::gen_block::{BlockID, GenBlock};
use crate::molecule::Molecule;
use crate::residue::Residue;
use crate::trajectory_frame_set::TrajectoryFrameSet;
use crate::{FRAME_DEPENDENT, MAX_STR_LEN, PARTICLE_DEPENDENT, utils};
use core::panic;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::{GzDecoder, ZlibDecoder};

/// Non trajectory blocks come before the first frame set block
#[derive(Debug)]
pub enum BlockTypeFlag {
    TrajectoryBlock,
    NonTrajectoryBlock,
}

/// Possible formats of data block contents
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DataType {
    #[default]
    Char = 0,
    Int = 1,
    Float = 2,
    Double = 3,
}

impl DataType {
    /// Try to interpret a raw i64 as a [`DataType`]
    ///
    /// # Panic
    /// Panics on unknown data types
    pub fn from_u8(raw: u8) -> Self {
        match raw {
            0 => DataType::Char,
            1 => DataType::Int,
            2 => DataType::Float,
            3 => DataType::Double,
            _ => panic!("unknown data type"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct BlockMetaInfo {
    /// The datatype of the data block.
    pub datatype: DataType,
    /// The dependency (particle and/or frame dependent)
    pub dependency: u8,
    /// set to TRUE if data is not written every frame.
    pub sparse_data: u8,
    /// set to the number of values per frame of the data.
    pub n_values: i64,
    /// set to the ID of the codec used to compress the data.
    pub codec_id: Compression,
    /// set to the first frame with data (only relevant if sparse_data == TRUE)
    pub first_frame_with_data: i64,
    /// set to the writing interval of the data (1 if sparse_data == FALSE)
    pub stride_length: i64,
    pub n_frames: i64,
    /// set to the number of the first particle with data written in this block
    pub num_first_particle: i64,
    /// set to the number of particles in this data block.
    pub block_n_particles: i64,
    /// set to the compression multiplier.
    pub multiplier: f64,
}

#[derive(Debug)]
pub struct Trajectory {
    /// Path to the input trajectory file.
    pub input_file_path: PathBuf,
    /// Open handle to the input file (None until opened).
    pub input_file: Option<File>,
    /// Length (in bytes) of the input file.
    pub input_file_len: u64,

    /// Path to the output trajectory file (if any).
    pub output_file_path: PathBuf,
    /// Open handle to the output file (None until opened).
    pub output_file: Option<File>,

    /// Name of the program that produced this trajectory.
    pub first_program_name: String,
    /// Force field used in the simulation.
    pub forcefield_name: String,
    /// Name of the user who first ran the simulation.
    pub first_user_name: String,
    /// Name of the computer where the simulation ran.
    pub first_computer_name: String,
    /// PGP signature of the user creating the file.
    pub first_pgp_signature: String,

    /// Name of the program used for the last modifications.
    pub last_program_name: String,
    /// Name of the user who last modified the file.
    pub last_user_name: String,
    /// Name of the computer where the last edits were made.
    pub last_computer_name: String,
    /// PGP signature of the user who last modified the file.
    pub last_pgp_signature: String,

    /// Creation time of the file, in seconds since UNIX epoch.
    pub creation_time: i64,
    /// Exponent for the distance unit (e.g., –9 for nm, –10 for Å).
    pub distance_unit_exponential: i64,

    /// Flag indicating if the number of atoms can vary (grand‐canonical, etc.).
    pub var_num_atoms: bool,
    /// Number of frames in each frame set (helps with indexing).
    pub frame_set_n_frames: i64,
    /// Number of frame sets in a “medium” stride.
    pub medium_stride_length: i64,
    /// Number of frame sets in a “long” stride.
    pub long_stride_length: i64,
    /// Duration (in seconds) of one frame (can change per frame set).
    pub time_per_frame: f64,

    /// Number of distinct molecule types in this trajectory.
    pub n_molecules: i64,
    /// Vector of molecule definitions.
    pub molecules: Vec<Molecule>,
    /// Count of each molecule type (length = `n_molecules`).
    pub(crate) molecule_cnt_list: Vec<i64>,
    /// Total number of particles (or atoms). If variable, updated per frame set.
    pub n_particles: i64,

    /// File‐offset (in bytes) of the first trajectory frame set in the input.
    pub first_trajectory_frame_set_input_pos: i64,
    /// File‐offset (in bytes) of the first trajectory frame set in the output.
    pub first_trajectory_frame_set_output_pos: i64,
    /// File‐offset (in bytes) of the last trajectory frame set in the input.
    pub last_trajectory_frame_set_input_pos: i64,
    /// File‐offset (in bytes) of the last trajectory frame set in the output.
    pub last_trajectory_frame_set_output_pos: i64,

    /// Metadata for the currently active frame set.
    pub current_trajectory_frame_set: TrajectoryFrameSet,
    /// File‐offset (in bytes) of the current frame set in the input.
    pub current_trajectory_frame_set_input_file_pos: i64,
    /// File‐offset (in bytes) of the current frame set in the output.
    pub current_trajectory_frame_set_output_file_pos: i64,
    /// Number of trajectory frame sets (not stored on disk; may be stale).
    pub n_trajectory_frame_sets: i64,

    /// Number of non‐frame‐dependent, particle‐dependent data blocks.
    pub n_particle_data_blocks: usize,
    /// Vector of particle‐dependent data blocks.
    pub non_tr_particle_data: Vec<Data>,

    /// Number of frame‐ and particle‐independent data blocks.
    pub n_data_blocks: usize,
    /// Vector of frame‐ and particle‐independent data blocks.
    pub non_tr_data: Vec<Data>,

    /// TNG compression algorithm for compressing positions
    pub compress_algo_pos: Vec<i32>,
    /// TNG compression algorithm for compressing velocities
    pub compress_algo_vel: Vec<i32>,
    /// Precision used for lossy compression.
    pub compression_precision: f64,
}

impl Trajectory {
    // TODO: do we need to check the endianness of the computer - perhaps?
    pub fn new() -> Self {
        Trajectory {
            input_file_path: PathBuf::new(),
            input_file: None,
            input_file_len: 0,

            output_file_path: PathBuf::new(),
            output_file: None,

            first_program_name: String::new(),
            forcefield_name: String::new(),
            first_user_name: String::new(),
            first_computer_name: String::new(),
            first_pgp_signature: String::new(),

            last_program_name: String::new(),
            last_user_name: String::new(),
            last_computer_name: String::new(),
            last_pgp_signature: String::new(),

            creation_time: 0,
            distance_unit_exponential: -9, // defaulting to nm

            var_num_atoms: false,
            frame_set_n_frames: 100,
            medium_stride_length: 100,
            long_stride_length: 10000,
            time_per_frame: -1.0,

            n_molecules: 0,
            molecules: Vec::new(),
            molecule_cnt_list: Vec::new(),
            n_particles: 0,

            first_trajectory_frame_set_input_pos: 0,
            first_trajectory_frame_set_output_pos: 0,
            last_trajectory_frame_set_input_pos: 0,
            last_trajectory_frame_set_output_pos: 0,

            current_trajectory_frame_set: TrajectoryFrameSet::new(),
            current_trajectory_frame_set_input_file_pos: 0,
            current_trajectory_frame_set_output_file_pos: 0,
            n_trajectory_frame_sets: 0,

            n_particle_data_blocks: 0,
            non_tr_particle_data: Vec::new(),

            n_data_blocks: 0,
            non_tr_data: Vec::new(),

            compress_algo_pos: Vec::new(),
            compress_algo_vel: Vec::new(),
            compression_precision: 1000.0,
        }
    }

    // c function: tng_input_file_set
    pub fn set_input_file(&mut self, path: &Path) {
        if self.input_file_path == path {
            return;
        }

        // If a file was already open, drop (close) it.
        if self.input_file.is_some() {
            self.input_file = None; // File is closed when dropped.
        }

        let truncated = if path.to_str().expect("valid unicode path").len() + 1 > MAX_STR_LEN {
            &path.to_str().unwrap()[..MAX_STR_LEN - 1]
        } else {
            path.to_str().unwrap()
        };

        // Allocate a new String. In Rust, this will panic on OOM by default.
        let new_path = truncated.to_string();

        // Assign it. Any previous String is dropped automatically.
        self.input_file_path = PathBuf::from(new_path);

        self.input_file_init();
    }

    pub fn input_file_init(&mut self) {
        if self.input_file.is_none() {
            // If no path has ever been set, error out
            if !self.input_file_path.exists() {
                eprintln!("No file specified for reading. {}:{}", file!(), line!());
                panic!();
            }

            // Try to open the file in "rb" mode (read‐only, binary)
            let path = self.input_file_path.clone();
            match File::open(&path) {
                Ok(f) => {
                    self.input_file = Some(f);
                }
                Err(_) => {
                    eprintln!(
                        "TNG library: Cannot open file {}. {}:{}",
                        path.display(),
                        file!(),
                        line!()
                    );
                    panic!();
                }
            }
        }

        // If we haven’t recorded the file’s length yet (i.e. it's still zero)…
        if self.input_file_len == 0 {
            let file_metadata = self
                .input_file
                .as_ref()
                .expect("input_file was available")
                .metadata();
            self.input_file_len = file_metadata.expect("the file to have metadata").len();
        }
    }

    // TODO: maybe these two go on GenBlock
    fn block_header_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();

        let start_pos = self
            .input_file
            .as_mut()
            .expect("we just init input_file")
            .stream_position()
            .expect("no error handling");

        block.header_contents_size =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&block.header_contents_size);

        if block.header_contents_size == 0 {
            block.id = BlockID::Unknown(0);
            warn!("header_contents_size was 0 block.id is Unknown(0)");
            return;
        }

        block.block_contents_size =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&block.block_contents_size);

        block.id = BlockID::from_u64(utils::read_u64_le_bytes(
            self.input_file.as_mut().expect("init input_file"),
        ));
        dbg!(&block.id);

        self.input_file
            .as_mut()
            .expect("we just init file")
            .read_exact(&mut block.md5_hash)
            .expect("no error handling");
        dbg!(block.md5_hash);

        block.name = Some(utils::fread_str(
            self.input_file.as_mut().expect("init input_file"),
        ));
        dbg!(&block.name);

        block.version =
            utils::read_u64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&block.version);

        let new_pos = (start_pos as i128 + block.header_contents_size as i128)
            .try_into()
            .expect("set new position when reading block header");
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            .expect("no error handling");
    }

    fn frame_set_block_read(&mut self, block: &mut GenBlock) {
        dbg!("frame_set_block_read");
        unreachable!();
    }
    fn trajectory_mapping_block_read(&mut self, block: &mut GenBlock) {
        dbg!("trajectory_mapping_block");
        unreachable!();
    }
    fn general_info_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();

        let start_pos = self
            .input_file
            .as_mut()
            .expect("we just init input_file")
            .stream_position()
            .expect("no error handling");
        let inp_file = self.input_file.as_mut().expect("init input_file");

        self.first_program_name = utils::fread_str(inp_file);
        self.last_program_name = utils::fread_str(inp_file);
        self.first_user_name = utils::fread_str(inp_file);
        self.last_user_name = utils::fread_str(inp_file);
        self.first_computer_name = utils::fread_str(inp_file);
        self.last_computer_name = utils::fread_str(inp_file);
        self.first_pgp_signature = utils::fread_str(inp_file);
        self.last_pgp_signature = utils::fread_str(inp_file);
        self.forcefield_name = utils::fread_str(inp_file);

        self.creation_time =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.creation_time);
        self.var_num_atoms =
            utils::read_bool_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.var_num_atoms);
        self.frame_set_n_frames =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.frame_set_n_frames);
        self.first_trajectory_frame_set_input_pos =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.first_trajectory_frame_set_input_pos);

        self.current_trajectory_frame_set.next_frame_set_file_pos =
            self.first_trajectory_frame_set_input_pos;
        self.last_trajectory_frame_set_input_pos =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.last_trajectory_frame_set_input_pos);

        self.medium_stride_length =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.medium_stride_length);

        self.long_stride_length =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.long_stride_length);

        if block.version >= 3 {
            self.distance_unit_exponential =
                utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
            dbg!(&self.distance_unit_exponential);
        }

        // TODO: Handle MD5 hashing here
        let new_pos = (start_pos as i128 + block.block_contents_size as i128)
            .try_into()
            .expect("set new position when reading block header");
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            .expect("no error handling");
    }

    fn molecules_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();
        let start_pos = self
            .input_file
            .as_mut()
            .expect("we just init input_file")
            .stream_position()
            .expect("no error handling");

        self.molecules.clear();

        self.n_molecules =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));
        dbg!(&self.n_molecules);

        self.n_particles = 0;
        self.molecules = Vec::with_capacity(self.n_molecules as usize);

        if !self.var_num_atoms {
            self.molecule_cnt_list = Vec::with_capacity(self.n_molecules as usize);
        }

        // Read each molecule from file
        for mol_idx in 0..self.n_molecules {
            let inp_file = self.input_file.as_mut().expect("init input_file");
            let mut molecule = Molecule::new();
            molecule.id = utils::read_i64_le_bytes(inp_file);
            dbg!(&molecule.id);
            molecule.name = utils::fread_str(inp_file);
            molecule.quaternary_str = utils::read_i64_le_bytes(inp_file);

            if !self.var_num_atoms {
                let count = utils::read_i64_le_bytes(inp_file);
                self.molecule_cnt_list.push(count);
            }

            molecule.n_chains = utils::read_i64_le_bytes(inp_file);
            molecule.n_residues = utils::read_i64_le_bytes(inp_file);
            molecule.n_atoms = utils::read_i64_le_bytes(inp_file);

            self.n_particles += molecule.n_atoms
                * self.molecule_cnt_list
                    [usize::try_from(mol_idx).expect("idx to molecule_cnt_list")];

            if molecule.n_chains > 0 {
                molecule.chains = Vec::with_capacity(molecule.n_chains as usize);
            }

            if molecule.n_residues > 0 {
                molecule.residues = Vec::with_capacity(molecule.n_residues as usize);
            }

            if molecule.n_atoms > 0 {
                molecule.atoms = Vec::with_capacity(molecule.n_atoms as usize);
            }

            // index counters to track positions in the flat `residues` and `atoms` vectors
            let mut residue_idx = 0;
            let mut atom_idx = 0;

            // Read the chains of the molecule
            for chain_idx in 0..molecule.n_chains {
                dbg!("starting chain loop");
                let mut chain = Chain::new();

                // Link back to parent molecule index
                chain.parent_molecule_idx = chain_idx as usize;
                chain.name = String::new();

                chain.read_data(self);

                // Determine this chain’s slice of `self.residues`:
                let start = residue_idx;
                let end = start + chain.n_residues;
                chain.residues_indices = (start as usize, end as usize);
                residue_idx = end; // next free residue slot

                dbg!(&chain);
                // Read the residues of the chain
                for local_idx in start..end {
                    dbg!("starting residue loop");
                    // let residue = &mut molecule.residues[local_idx as usize];
                    let mut residue = Residue::new();

                    // Link back to parent chain index
                    residue.chain_index = Some(chain_idx as usize);
                    residue.name = String::new();

                    residue.read_data(self);

                    // Compute atoms_offset = `atom - molecule->atoms` in C
                    residue.atoms_offset = atom_idx;

                    dbg!(&residue);
                    let atom_count = residue.n_atoms;

                    // Read the atoms of the residue
                    for _ in 0..atom_count {
                        let mut atom = Atom::new();

                        // Link back to parent residue index
                        residue.chain_index = Some(chain_idx as usize);
                        atom.residue_index = Some(local_idx);

                        atom.read_data(self);

                        atom_idx += 1;
                        dbg!(&atom);
                        molecule.atoms.push(atom);
                    }
                    molecule.residues.push(residue);
                }
                molecule.chains.push(chain);
            }

            // If no chains but there *are* residues (i.e., n_chains == 0 && n_residues > 0):
            if molecule.n_chains == 0 && molecule.n_residues > 0 {
                for r_index in 0..molecule.n_residues {
                    let residue = &mut molecule.residues[r_index as usize];

                    // Link to no chain: `residue->chain = 0;`
                    residue.chain_index = None;
                    residue.name = String::new();

                    residue.read_data(self);

                    residue.atoms_offset = atom_idx;
                    let atom_count = residue.n_atoms;
                    for _ in 0..atom_count {
                        let mut atom = Atom::new();

                        atom.residue_index = Some(r_index as u64);
                        atom.read_data(self);
                        atom_idx += 1;
                    }
                }
            }

            // If no chains and no residues, read atoms directly:
            if molecule.n_chains == 0 && molecule.n_residues == 0 {
                for _ in 0..molecule.n_atoms {
                    let mut atom = Atom::new();
                    atom.residue_index = None;
                    atom.read_data(self);
                }
            }

            let inp_file = self.input_file.as_mut().expect("init input_file");
            molecule.n_bonds = utils::read_i64_le_bytes(inp_file);

            for _ in 0..molecule.n_bonds {
                let mut bond = Bond::new();
                bond.from_atom_id = utils::read_i64_le_bytes(inp_file);
                bond.from_atom_id = utils::read_i64_le_bytes(inp_file);
                molecule.bonds.push(bond);
            }

            self.molecules.push(molecule);
        }

        let new_pos = (start_pos as i128 + block.block_contents_size as i128)
            .try_into()
            .expect("set new position when reading block header");
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            .expect("no error handling");
    }

    /// Read the meta information of a data block (particle or non-particle data).
    fn data_block_meta_information_read(&mut self, block: &mut GenBlock) -> BlockMetaInfo {
        let mut block_meta_info = BlockMetaInfo::default();
        let inp_file = self.input_file.as_mut().expect("init input_file");

        block_meta_info.datatype = DataType::from_u8(utils::read_u8(inp_file));
        block_meta_info.dependency = utils::read_u8(inp_file);

        if block_meta_info.dependency & FRAME_DEPENDENT != 0 {
            block_meta_info.sparse_data = utils::read_u8(inp_file);
        }
        block_meta_info.n_values = utils::read_i64_le_bytes(inp_file);
        block_meta_info.codec_id = Compression::from_i64(utils::read_i64_le_bytes(inp_file));

        block_meta_info.multiplier = if block_meta_info.codec_id != Compression::Uncompressed {
            dbg!("uncompressed");
            utils::read_f64_bytes(inp_file)
        } else {
            1.0
        };

        if block_meta_info.dependency & FRAME_DEPENDENT != 0 {
            if block_meta_info.sparse_data != 0 {
                block_meta_info.first_frame_with_data = utils::read_i64_le_bytes(inp_file);
                block_meta_info.stride_length = utils::read_i64_le_bytes(inp_file);
                block_meta_info.n_frames = self.current_trajectory_frame_set.n_frames
                    - (block_meta_info.first_frame_with_data
                        - self.current_trajectory_frame_set.first_frame);
            } else {
                block_meta_info.first_frame_with_data =
                    self.current_trajectory_frame_set.first_frame;
                block_meta_info.stride_length = 1;
                block_meta_info.n_frames = self.current_trajectory_frame_set.n_frames;
            }
        } else {
            block_meta_info.first_frame_with_data = 0;
            block_meta_info.stride_length = 1;
            block_meta_info.n_frames = 1;
        }

        if block_meta_info.dependency & PARTICLE_DEPENDENT != 0 {
            block_meta_info.num_first_particle = utils::read_i64_le_bytes(inp_file);
            block_meta_info.block_n_particles = utils::read_i64_le_bytes(inp_file);
        } else {
            block_meta_info.num_first_particle = -1;
            block_meta_info.block_n_particles = 0;
        }

        dbg!(&block_meta_info);
        block_meta_info
    }

    fn particle_data_find(&mut self, id: BlockID) -> Option<Data> {
        let is_traj_block = self.current_trajectory_frame_set_input_file_pos > 0
            || self.current_trajectory_frame_set_output_file_pos > 0;

        let mut block_index = -1;
        if is_traj_block {
            // Search in frame_set.tr_particle_data
            let frame_set = &self.current_trajectory_frame_set;
            assert_eq!(
                frame_set.n_particle_data_blocks,
                frame_set.tr_particle_data.len()
            );
            for block in &frame_set.tr_particle_data {
                if block.block_id == id {
                    return Some(block.clone());
                }
            }
        } else {
            // Search in self.non_tr_particle_data
            assert_eq!(self.n_particle_data_blocks, self.non_tr_particle_data.len());
            for block in &self.non_tr_particle_data {
                if block.block_id == id {
                    return Some(block.clone());
                }
            }
        }
        None
    }

    fn data_find(&mut self, id: BlockID) -> Option<Data> {
        // Determine whether we should search trajectory‐block data
        let is_traj_block = self.current_trajectory_frame_set_input_file_pos > 0
            || self.current_trajectory_frame_set_output_file_pos > 0;

        let frame_set = &self.current_trajectory_frame_set;

        if is_traj_block {
            // Search in frame_set.tr_data
            assert_eq!(frame_set.n_data_blocks, frame_set.tr_data.len());
            for block in &frame_set.tr_data {
                if block.block_id == id {
                    return Some(block.clone());
                }
            }
            // If not found there, fall back to non_tr_data
            assert_eq!(self.n_data_blocks, self.non_tr_data.len());
            for block in &self.non_tr_data {
                if block.block_id == id {
                    return Some(block.clone());
                }
            }
        } else {
            // Only search non_tr_data
            assert_eq!(self.n_data_blocks, self.non_tr_data.len());
            for block in &self.non_tr_data {
                if block.block_id == id {
                    return Some(block.clone());
                }
            }
        }

        None
    }

    fn particle_data_block_create(&mut self, is_traj_block: bool) {
        let frame_set = &mut self.current_trajectory_frame_set;
        if is_traj_block {
            frame_set.n_particle_data_blocks += 1;
            frame_set.tr_particle_data.push(Data::default());
        } else {
            self.n_particle_data_blocks += 1;
            self.non_tr_particle_data.push(Data::default());
        }
    }

    fn data_block_create(&mut self, is_traj_block: bool) {
        let frame_set = &mut self.current_trajectory_frame_set;
        if is_traj_block {
            frame_set.n_data_blocks += 1;
            frame_set.tr_data.push(Data::default());
        } else {
            self.n_data_blocks += 1;
            self.non_tr_data.push(Data::default());
        }
    }

    fn gzip_uncompress(data: &[u8], compressed_len: u64, uncompressed_len: usize) -> Vec<u8> {
        let mut output = vec![0u8; uncompressed_len];

        let cursor = &data[..compressed_len as usize];
        let mut decoder = ZlibDecoder::new(cursor);
        let mut reader = decoder.take(uncompressed_len as u64);
        match reader.read(&mut output) {
            Ok(bytes_read) => {
                if bytes_read != uncompressed_len {
                    // C’s `uncompress` updates new_len to the actual decompressed size.
                    // If it doesn’t match the expected `uncompressed_len`, that’s an error.
                    eprintln!(
                        "TNG library: Expected {} bytes, but uncompressed {} bytes.\n",
                        uncompressed_len, bytes_read
                    );
                    panic!();
                }
                // Drop the old buffer and replace it with `dest`.
                output
            }
            Err(e) => {
                // Map common I/O errors to C’s zlib error messages:
                // - UnexpectedEof  → buffer too small (Z_BUF_ERROR)
                // - InvalidData    → data corrupt (Z_DATA_ERROR)
                // - Other I/O errs → generic uncompress error
                panic!();
                // match e.kind() {
                //     io::ErrorKind::UnexpectedEof => {
                //         eprintln!("TNG library: Destination buffer too small. ");
                //     }
                //     io::ErrorKind::InvalidData => {
                //         eprintln!("TNG library: Data corrupt. ");
                //     }
                //     _ => {
                //         eprintln!("TNG library: Error uncompressing gzipped data. ");
                //     }
                // }
                // TngStatus::Failure
            }
        }
    }

    /// Read the values of a data block
    /// c function name: tng_data_read
    fn data_read(&mut self, block: &mut GenBlock, meta_info: BlockMetaInfo, block_data_len: u64) {
        // we pull what we need early from the current_trajectory_frame_set to avoid the borrow checker
        let frame_set_n_particles = self.current_trajectory_frame_set.n_particles;

        let size = match meta_info.datatype {
            DataType::Char => 1,
            DataType::Int => size_of::<i64>(),
            DataType::Float => size_of::<f32>(),
            DataType::Double => size_of::<f64>(),
        };

        let is_particle_data = if meta_info.block_n_particles > 0 {
            true
        } else {
            if meta_info.codec_id == Compression::XTC || meta_info.codec_id == Compression::TNG {
                eprintln!("No file specified for reading. {}:{}", file!(), line!());
                panic!();
            }
            false
        };

        let maybe_data = if is_particle_data {
            &mut self.particle_data_find(block.id)
        } else {
            &mut self.data_find(block.id)
        };

        let is_traj_block = self.current_trajectory_frame_set_input_file_pos > 0;

        // If the block does not exist, create it
        let data = if let Some(existing) = maybe_data {
            existing
        } else {
            dbg!("data block did not exist");
            if is_particle_data {
                self.particle_data_block_create(is_traj_block);
            } else {
                self.data_block_create(is_traj_block);
            }

            let frame_set = &mut self.current_trajectory_frame_set;
            dbg!(&frame_set);
            let data = if is_particle_data {
                if is_traj_block {
                    frame_set
                        .tr_particle_data
                        .last_mut()
                        .expect("available tr_particle_data")
                } else {
                    self.non_tr_particle_data
                        .last_mut()
                        .expect("available element on non_tr_particle_data")
                }
            } else if is_traj_block {
                frame_set
                    .tr_data
                    .last_mut()
                    .expect("available element on tr_data")
            } else {
                self.non_tr_data
                    .last_mut()
                    .expect("available element on non_tr_data")
            };
            data.block_id = block.id;
            data.block_name = block.name.as_ref().expect("block to have a name").clone();
            data.data_type = meta_info.datatype;
            data.values = None;

            // from c - FIXME: Memory leak from strings
            data.strings = None;
            data.n_frames = 0;
            data.dependency = 0;
            if is_particle_data {
                data.dependency |= PARTICLE_DEPENDENT;
            }

            if is_traj_block
                && (meta_info.n_frames > 1
                    || frame_set.n_frames == meta_info.n_frames
                    || meta_info.stride_length > 1)
            {
                data.dependency |= FRAME_DEPENDENT;
            }
            data.codec_id = meta_info.codec_id;
            data.compression_multiplier = meta_info.multiplier;
            data.last_retrieved_frame = -1;

            data
        };

        let tot_n_particles = if is_particle_data {
            if is_traj_block && self.var_num_atoms {
                frame_set_n_particles
            } else {
                self.n_particles
            }
        } else {
            1
        };

        let n_frames_div = (meta_info.n_frames - 1) / meta_info.stride_length + 1;
        let mut contents = vec![0; usize::try_from(block_data_len).expect("u64 to usize")];
        if self
            .input_file
            .as_mut()
            .expect("init input_file")
            .read_exact(&mut contents)
            .is_err()
        {
            eprintln!("Cannot read block. {}:{}", file!(), line!());
            panic!();
        }

        // TODO: hash mode
        if data.codec_id != Compression::Uncompressed {
            let mut full_data_len = (n_frames_div as usize)
                .checked_mul(size)
                .and_then(|x| x.checked_mul(meta_info.n_values as usize))
                .unwrap_or(0);
            if is_particle_data {
                full_data_len = full_data_len
                    .checked_mul(meta_info.block_n_particles as usize)
                    .expect("mul of meta_info.block_n_particles");
            }

            let mut actual_contents = Vec::new();
            match data.codec_id {
                Compression::Uncompressed => {
                    full_data_len = usize::try_from(block_data_len).expect("usize from u64")
                }
                Compression::XTC => todo!("XTC compression not implemented yet"),
                Compression::TNG => todo!("TNG is todo"),
                Compression::GZip => {
                    dbg!("from gzip: ", full_data_len);
                    println!("before compression {}", block.block_contents_size);
                    actual_contents =
                        Trajectory::gzip_uncompress(&contents, block_data_len, full_data_len);
                    println!("after compression {}", block.block_contents_size);
                }
            }

            // Allocate memory
            // we assume that data.values is always allocated, but may be None. C code did something like
            // !data->values
            if data.values.is_none()
                || data.n_frames != meta_info.n_frames
                || data.n_values_per_frame != meta_info.n_values
            {
                println!("data.values was None so we allocate");
                if is_particle_data {
                    data.allocate_particle_data_mem(
                        meta_info.n_frames,
                        meta_info.stride_length,
                        tot_n_particles,
                        meta_info.n_values,
                    )
                } else {
                    data.allocate_data_mem(
                        meta_info.n_frames,
                        meta_info.stride_length,
                        meta_info.n_values,
                    )
                };
            }
            data.first_frame_with_data = meta_info.first_frame_with_data;

            if meta_info.datatype == DataType::Char {
                // We expect `strings` to be Some(…) and shape at least [n_frames_div][…][…].
                let strings_3d = match &mut data.strings {
                    Some(s) => s,
                    None => unreachable!("data.strings was None"),
                };
                let mut offset = 0;
                // Strings are stored slightly differently if the data block contains
                // particle data (frames * particles * n_values) or not (frames * n_values)
                if is_particle_data {
                    for i in 0..n_frames_div {
                        // Get the Vec<Vec<String>> for this frame
                        let first_dim_values = &mut strings_3d[i as usize];

                        for j in meta_info.num_first_particle
                            ..meta_info.num_first_particle + self.n_particles
                        {
                            let second_dim_values = &mut first_dim_values[j as usize];
                            for k in 0..meta_info.n_values {
                                // Find the length of the C‐string at `contents[offset..]`, capped by TNG_MAX_STR_LEN
                                let remaining = &actual_contents[offset..];
                                let nul_position = remaining
                                    .iter()
                                    .position(|&b| b == 0)
                                    .unwrap_or(MAX_STR_LEN - 1);
                                // length of this C‐string including NUL
                                let raw_len = (nul_position + 1).min(MAX_STR_LEN);

                                // Extract the bytes before the NUL (i.e. [offset .. offset + raw_len - 1])
                                if offset + raw_len > actual_contents.len() {
                                    panic!("ran out of bounds")
                                }
                                let str_bytes = &actual_contents[offset..offset + raw_len - 1];

                                let s = String::from_utf8_lossy(str_bytes).into_owned();

                                // Store/overwrite into `strings[frame_idx][particle_idx][val_idx]`
                                second_dim_values[k as usize] = s;

                                // Advance offset by raw_len (skip the NUL too)
                                offset += raw_len;
                            }
                        }
                    }
                } else {
                    for i in 0..n_frames_div {
                        for j in 0..meta_info.n_values {
                            // Find the length of the C‐string at `contents[offset..]`, capped by TNG_MAX_STR_LEN
                            let remaining = &actual_contents[offset..];
                            let nul_position = remaining
                                .iter()
                                .position(|&b| b == 0)
                                .unwrap_or(MAX_STR_LEN - 1);
                            // length of this C‐string including NUL
                            let raw_len = (nul_position + 1).min(MAX_STR_LEN);

                            // Extract the bytes before the NUL (i.e. [offset .. offset + raw_len - 1])
                            if offset + raw_len > actual_contents.len() {
                                panic!("ran out of bounds")
                            }
                            let str_bytes = &actual_contents[offset..offset + raw_len - 1];

                            let s = String::from_utf8_lossy(str_bytes).into_owned();

                            // Store/overwrite into `strings[0][particle_idx][val_idx]`
                            strings_3d[0][i as usize][j as usize] = s;

                            // Advance offset by raw_len (skip the NUL too)
                            offset += raw_len;
                        }
                    }
                }
            } else {
                if is_particle_data {
                    // Compute the byte‐offset: n_frames_div * size * n_values * num_first_particle
                    let offset = usize::try_from(
                        n_frames_div
                            .checked_mul(size as i64)
                            .and_then(|v| v.checked_mul(meta_info.n_values))
                            .and_then(|v| v.checked_mul(meta_info.num_first_particle))
                            .expect("offset overflow"),
                    )
                    .expect("i64 to usize");
                    dbg!(&offset);

                    dbg!(&full_data_len);
                    dbg!(&n_frames_div);
                    dbg!(&size);
                    dbg!(&meta_info.n_values);
                    dbg!(&meta_info.num_first_particle);
                    data.values.as_mut().expect("data.values to be Some")
                        [offset..offset + full_data_len]
                        .copy_from_slice(&actual_contents[..full_data_len]);
                } else {
                    data.values.as_mut().expect("data.values to be Some")[..full_data_len]
                        .copy_from_slice(&actual_contents[..full_data_len]);
                }

                // TODO: handle endianness here
                if data.codec_id != Compression::TNG {
                    match data.data_type {
                        DataType::Float => {}
                        DataType::Int => {}
                        DataType::Double => {}
                        DataType::Char => {}
                    }
                }
            }
            dbg!(&data.values.as_ref().expect("something")[..10]);
        }
    }

    /// Read the contents of a data block (particle or non-particle data)
    fn data_block_contents_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();
        let start_pos = self
            .input_file
            .as_mut()
            .expect("we just init input_file")
            .stream_position()
            .expect("no error handling");
        dbg!(start_pos);

        let meta_info = self.data_block_meta_information_read(block);

        let current_pos = self
            .input_file
            .as_mut()
            .expect("we just init input_file")
            .stream_position()
            .expect("no error handling");
        let remaining_len = block.block_contents_size as u64 - (current_pos - start_pos);

        self.data_read(block, meta_info, remaining_len);

        // TODO: handle md5 hash

        // if hash_mode == TNG_USE_HASH {}

        let new_pos = start_pos + block.block_contents_size as u64;
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            .expect("no error handling");
    }

    fn block_read_next(&mut self, block: &mut GenBlock) {
        match block.id {
            BlockID::TrajectoryFrameSet => self.frame_set_block_read(block),
            BlockID::ParticleMapping => self.trajectory_mapping_block_read(block),
            BlockID::GeneralInfo => self.general_info_block_read(block),
            BlockID::Molecules => self.molecules_block_read(block),
            // id if id >= BlockID::TrajBoxShape => self.data_block_contents_read(block),
            id => {
                if id >= BlockID::TrajBoxShape {
                    self.data_block_contents_read(block);
                } else {
                    // We skip to the next block
                    let current_pos = self
                        .input_file
                        .as_mut()
                        .expect("we just init input_file")
                        .stream_position()
                        .expect("no error handling");
                    let new_pos = (current_pos as i128 + block.block_contents_size as i128)
                        .try_into()
                        .expect("set new position when reading block header");
                    self.input_file
                        .as_mut()
                        .expect("init input_file")
                        .seek(SeekFrom::Start(new_pos))
                        .expect("no error handling");
                }
            }
        }
    }

    pub fn file_headers_read(&mut self) {
        if self.input_file.is_some() {
            self.n_trajectory_frame_sets = 0;

            // TODO: do we need to call this here? aren't we guaranteed to
            // have init the input file?
            self.input_file_init();

            let mut prev_pos: u64 = 0;
            let mut block = GenBlock::new();
            loop {
                if prev_pos >= self.input_file_len {
                    break;
                }
                self.block_header_read(&mut block);

                // If `block.id == -1` or `block.id == TNG_TRAJECTORY_FRAME_SET`, exit loop
                match block.id {
                    BlockID::Unknown(0) | BlockID::TrajectoryFrameSet => break,
                    _ => {}
                }

                println!("calling block_read_next");
                self.block_read_next(&mut block);
                prev_pos = self
                    .input_file
                    .as_mut()
                    .expect("we just init input_file")
                    .stream_position()
                    .expect("no error handling");
            }

            if block.id == BlockID::TrajectoryFrameSet {
                self.input_file
                    .as_mut()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(prev_pos))
                    .expect("no error handling");
            }
        }
    }

    /// Find a molecule by name and/or ID.
    ///
    /// # Parameters
    ///
    /// - `name`: The name of the molecule to search for. If this is an empty string,
    ///   only `id` is used.
    /// - `id`: The numeric ID of the molecule to search for. If this is `-1`, only `name`
    ///   is used.
    pub fn find_molecule(&self, name: &str, id: i64) -> Option<&Molecule> {
        if name.is_empty() && id == -1 {
            // Return first if available
            // return self.molecules.first().ok_or(MoleculeFindError::NotFound);
            return self.molecules.first();
        }

        if !name.is_empty() && id != -1 {
            // Both name and id must match
            for mol in &self.molecules {
                if mol.name == name && mol.id == id {
                    return Some(mol);
                }
            }
        } else if !name.is_empty() {
            // Only name match
            for mol in &self.molecules {
                if mol.name == name {
                    return Some(mol);
                }
            }
        } else {
            // Only id match
            for mol in &self.molecules {
                if mol.id == id {
                    return Some(mol);
                }
            }
        }

        None

        // Err(MoleculeFindError::NotFound)
    }

    /// Get the list of the count of each molecule
    pub fn molecule_cnt_list_get(&self) -> Vec<i64> {
        if self.var_num_atoms {
            self.current_trajectory_frame_set.molecule_cnt_list.clone()
        } else {
            self.molecule_cnt_list.clone()
        }
    }

    pub fn molecule_id_of_particle_nr_get(&self, nr: i64) -> Option<i64> {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            return Some(mol.id);
        }
        None
    }
}
