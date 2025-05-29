use log::warn;
use std::cmp::max;

use crate::atom::Atom;
use crate::bond::Bond;
use crate::chain::Chain;
use crate::data::{Compression, Data};
use crate::gen_block::{BlockID, GenBlock};
use crate::molecule::Molecule;
use crate::particle_mapping::ParticleMapping;
use crate::residue::Residue;
use crate::trajectory_frame_set::TrajectoryFrameSet;
use crate::{FRAME_DEPENDENT, MAX_STR_LEN, PARTICLE_DEPENDENT, utils};
use core::panic;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;

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

    fn get_size(&self) -> usize {
        match self {
            DataType::Char => 1,
            DataType::Int => size_of::<i64>(),
            DataType::Float => size_of::<f32>(),
            DataType::Double => size_of::<f64>(),
        }
    }
}

fn is_same_file(file1: &File, file2: &File) -> std::io::Result<bool> {
    let meta1 = file1.metadata()?;
    let meta2 = file2.metadata()?;

    Ok(meta1.ino() == meta2.ino() && meta1.dev() == meta2.dev())
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

    fn get_file_position(&mut self) -> u64 {
        self.input_file
            .as_ref()
            .expect("init input_file")
            .stream_position()
            .expect("no error handling")
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

        let start_pos = self.get_file_position();

        block.header_contents_size =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));

        if block.header_contents_size == 0 {
            block.id = BlockID::Unknown(0);
            warn!("header_contents_size was 0 block.id is Unknown(0)");
            return;
        }

        block.block_contents_size =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));

        block.id = BlockID::from_u64(utils::read_u64_le_bytes(
            self.input_file.as_mut().expect("init input_file"),
        ));

        self.input_file
            .as_ref()
            .expect("we just init file")
            .read_exact(&mut block.md5_hash)
            .expect("no error handling");

        block.name = Some(utils::fread_str(
            self.input_file.as_mut().expect("init input_file"),
        ));

        block.version =
            utils::read_u64_le_bytes(self.input_file.as_mut().expect("init input_file"));

        let new_pos = (start_pos as i128 + block.header_contents_size as i128)
            .try_into()
            .expect("set new position when reading block header");
        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            .expect("no error handling");
    }

    fn frame_set_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();
        let start_pos = self.get_file_position();

        // FIXME (from c): Does not check if the size of the contents matches the
        // expected size of if the contents can be read
        let file_pos = start_pos - u64::try_from(block.header_contents_size).expect("i64 to u64");
        self.current_trajectory_frame_set_input_file_pos =
            i64::try_from(file_pos).expect("u64 to i64");
        // set_particle_mapping_free
        let frame_set = &mut self.current_trajectory_frame_set;
        let inp_file = self.input_file.as_mut().expect("init input_file");
        frame_set.first_frame = utils::read_i64_le_bytes(inp_file);
        frame_set.n_frames = utils::read_i64_le_bytes(inp_file);

        if self.var_num_atoms {
            // let prev_n_particles = frame_set.n_particles;
            frame_set.n_particles = 0;

            for (mol, mol_count) in self
                .molecules
                .iter()
                .zip(frame_set.molecule_cnt_list.iter_mut())
            {
                *mol_count = utils::read_i64_le_bytes(inp_file);
                frame_set.n_particles += mol.n_atoms * *mol_count;
            }

            // from the c code
            // if prev_n_particles && frame_set.n_particles != prev_n_particles {
            //     /* FIXME: Particle dependent data memory management */
            // }
        }

        frame_set.next_frame_set_file_pos = utils::read_i64_le_bytes(inp_file);
        frame_set.prev_frame_set_file_pos = utils::read_i64_le_bytes(inp_file);
        frame_set.medium_stride_next_frame_set_file_pos = utils::read_i64_le_bytes(inp_file);
        frame_set.medium_stride_prev_frame_set_file_pos = utils::read_i64_le_bytes(inp_file);
        frame_set.long_stride_next_frame_set_file_pos = utils::read_i64_le_bytes(inp_file);
        frame_set.long_stride_prev_frame_set_file_pos = utils::read_i64_le_bytes(inp_file);

        if block.version >= 3 {
            frame_set.first_frame_time = utils::read_f64_bytes(inp_file);
            self.time_per_frame = utils::read_f64_bytes(inp_file);
        } else {
            frame_set.first_frame_time = -1.0;
            self.time_per_frame = -1.0;
        }

        // TODO: Hash mode
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(
                start_pos + u64::try_from(block.block_contents_size).expect("u64 from i64"),
            ))
            .expect("no error handling");

        // If the output file and the input files are the same the number of
        // frames in the file are the same number as has just been read.
        // This is updated here to later on see if there have been new frames
        // added and thereby the frame set needs to be rewritten.
        if self.output_file.is_some()
            && self.input_file.is_some()
            && is_same_file(
                self.output_file.as_ref().expect("output file set"),
                self.input_file.as_ref().expect("input file set"),
            )
            .is_ok_and(|x| x)
        {
            frame_set.n_written_frames = frame_set.n_frames;
        }
    }

    fn trajectory_mapping_block_read(&mut self, block: &mut GenBlock) {
        dbg!("trajectory_mapping_block");
        self.input_file_init();

        let start_pos = self.get_file_position();
        let inp_file = self.input_file.as_mut().expect("init input_file");

        // FIXME (from c): Does not check if the size of the contents matches the
        // expected size of if the contents can be read
        let frame_set = &mut self.current_trajectory_frame_set;
        frame_set.n_mapping_blocks += 1;
        let mut mapping = ParticleMapping::new();

        // TODO: hash mode

        mapping.num_first_particle = utils::read_i64_le_bytes(inp_file);
        mapping.n_particles = utils::read_i64_le_bytes(inp_file);
        mapping.real_particle_numbers.resize(
            usize::try_from(mapping.n_particles).expect("i64 to usize"),
            0,
        );

        // TODO: handle endianness
        let bytes_to_read = usize::try_from(mapping.n_particles).expect("i64 to usize")
            * std::mem::size_of::<i64>();
        let mut buffer = vec![0u8; bytes_to_read];

        match inp_file.read_exact(&mut buffer) {
            Ok(()) => {
                // Convert bytes to i64 values safely
                for (i, chunk) in buffer.chunks_exact(8).enumerate() {
                    let bytes: [u8; 8] = chunk.try_into().expect("chunk is exactly 8 bytes");
                    mapping.real_particle_numbers[i] = i64::from_le_bytes(bytes);
                }

                // TODO: Handle hashing
            }
            Err(_) => {
                eprintln!("Cannot read block. {}:{}", file!(), line!());
                panic!()
            }
        }
        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(
                start_pos + u64::try_from(block.block_contents_size).expect("u64 from i64"),
            ))
            .expect("no error handling");
    }

    fn general_info_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();

        let start_pos = self.get_file_position();
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

        self.creation_time = utils::read_i64_le_bytes(inp_file);
        self.var_num_atoms = utils::read_bool_le_bytes(inp_file);
        self.frame_set_n_frames = utils::read_i64_le_bytes(inp_file);
        self.first_trajectory_frame_set_input_pos = utils::read_i64_le_bytes(inp_file);

        self.current_trajectory_frame_set.next_frame_set_file_pos =
            self.first_trajectory_frame_set_input_pos;
        self.last_trajectory_frame_set_input_pos = utils::read_i64_le_bytes(inp_file);

        self.medium_stride_length = utils::read_i64_le_bytes(inp_file);

        self.long_stride_length = utils::read_i64_le_bytes(inp_file);

        if block.version >= 3 {
            self.distance_unit_exponential = utils::read_i64_le_bytes(inp_file);
        }

        // TODO: Handle MD5 hashing here
        let new_pos = (start_pos as i128 + block.block_contents_size as i128)
            .try_into()
            .expect("set new position when reading block header");
        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            .expect("no error handling");
    }

    fn molecules_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();
        let start_pos = self.get_file_position();

        self.molecules.clear();

        self.n_molecules =
            utils::read_i64_le_bytes(self.input_file.as_mut().expect("init input_file"));

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
                        atom.residue_index =
                            Some(usize::try_from(local_idx).expect("local_idx to usize"));

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

                        atom.residue_index =
                            Some(usize::try_from(r_index).expect("r_index to usize"));
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

    fn particle_data_find(&self, id: BlockID) -> Option<Data> {
        let is_traj_block = self.current_trajectory_frame_set_input_file_pos > 0
            || self.current_trajectory_frame_set_output_file_pos > 0;

        // let block_index = -1;
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

    fn data_find(&self, id: BlockID) -> Option<Data> {
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

    fn gzip_uncompress(
        data: &[u8],
        compressed_len: u64,
        uncompressed_len: usize,
    ) -> Result<Vec<u8>, ()> {
        let mut output = vec![0u8; uncompressed_len];

        let cursor = &data[..compressed_len as usize];
        let mut decoder = ZlibDecoder::new(cursor);
        // let mut reader = decoder.take(uncompressed_len as u64);
        // match reader.read(&mut output) {
        match decoder.read_exact(&mut output) {
            Ok(()) => {
                // if bytes_read != uncompressed_len {
                //     // C’s `uncompress` updates new_len to the actual decompressed size.
                //     // If it doesn’t match the expected `uncompressed_len`, that’s an error.
                //     eprintln!(
                //         "Expected {} bytes, but uncompressed {} bytes.\n",
                //         uncompressed_len, bytes_read
                //     );
                //     panic!();
                // }
                // Drop the old buffer and replace it with `dest`.
                Ok(output)
            }
            Err(e) => {
                // Map common I/O errors to C’s zlib error messages:
                // - UnexpectedEof  → buffer too small (Z_BUF_ERROR)
                // - InvalidData    → data corrupt (Z_DATA_ERROR)
                // - Other I/O errs → generic uncompress error
                eprintln!("{}", e);
                return Err(());
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
    fn data_read(
        &mut self,
        block: &mut GenBlock,
        meta_info: BlockMetaInfo,
        block_data_len: u64,
    ) -> Result<(), ()> {
        // we pull what we need early from the current_trajectory_frame_set to avoid the borrow checker
        let frame_set_n_particles = self.current_trajectory_frame_set.n_particles;

        let size = meta_info.datatype.get_size();

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
                Compression::Uncompressed => {}
                Compression::XTC => todo!("XTC compression not implemented yet"),
                Compression::TNG => todo!("TNG is todo"),
                Compression::GZip => {
                    let uncompressed_result =
                        Trajectory::gzip_uncompress(&contents, block_data_len, full_data_len);
                    if uncompressed_result.is_ok() {
                        actual_contents = uncompressed_result.unwrap();
                    } else {
                        return Err(());
                    }
                }
            }

            // Allocate memory
            // we assume that data.values is always allocated, but may be None. C code did something like
            // !data->values
            if data.values.is_none()
                || data.n_frames != meta_info.n_frames
                || data.n_values_per_frame != meta_info.n_values
            {
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
        Ok(())
    }

    /// Read the contents of a data block (particle or non-particle data)
    fn data_block_contents_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();
        let start_pos = self.get_file_position();

        let meta_info = self.data_block_meta_information_read(block);

        let current_pos = self.get_file_position();
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

    /// Read one (the next) block (of any kind) from the input_file of [`Self`]
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
                    let current_pos = self.get_file_position();
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
                prev_pos = self.get_file_position();
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
    pub fn molecule_cnt_list_get(&self) -> &Vec<i64> {
        if self.var_num_atoms {
            &self.current_trajectory_frame_set.molecule_cnt_list
        } else {
            &self.molecule_cnt_list
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

    pub fn residue_id_of_particle_nr_get(&self, nr: i64) -> Option<i64> {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut atom_id = None;
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            atom_id = Some(mol.atoms[(nr % mol.n_atoms) as usize].id)
        }
        atom_id
    }

    /// Get the residue id (based on other molecules and molecule counts) of real
    /// particle number (number in the mol system)
    pub fn global_residue_id_of_particle_nr_get(&self, nr: i64) -> Option<u64> {
        let mut count = 0;
        let mut offset = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut atom_residue_index = None;
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                offset += mol.n_residues * mol_count;
                continue;
            }
            let residue_idx = mol.atoms[(nr % mol.n_atoms) as usize]
                .residue_index
                .expect("residue index");
            offset += mol.n_residues * ((nr - count) / mol.n_atoms);
            atom_residue_index = Some(mol.residues[residue_idx as usize].id + offset as u64);
        }
        atom_residue_index
    }

    /// Get the molecule name of real particle number (number in mol system)
    pub fn molecule_name_of_particle_nr_get(&self, nr: i64) -> String {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = String::new();
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            name = mol.name.clone();
        }
        name
    }

    /// Get the chain name of real particle number (number in mol system)
    pub fn chain_name_of_particle_nr_get(&self, nr: i64) -> String {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = String::new();
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            let atom = &mol.atoms[(nr % mol.n_atoms) as usize];
            let residue_index = atom.residue_index.expect("atom in residue");
            let chain_index = &mol.residues[residue_index]
                .chain_index
                .expect("residue in chain");
            name = mol.chains[*chain_index].name.clone();
        }
        name
    }

    /// Get the residue name of real particle number (number in mol system).
    pub fn residue_name_of_particle_nr_get(&self, nr: i64) -> String {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = String::new();
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            let atom = &mol.atoms[(nr % mol.n_atoms) as usize];
            let residue_index = atom.residue_index.expect("atom in residue");
            name = mol.residues[residue_index].name.clone();
        }
        name
    }

    /// Get the atom name of real particle number (number in mol system).
    pub fn atom_name_of_particle_nr_get(&self, nr: i64) -> String {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = String::new();
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            let atom = &mol.atoms[(nr % mol.n_atoms) as usize];
            name = atom.name.clone();
        }
        name
    }

    /// Add an existing [`Molecule`] to [`Self`]
    pub fn molecule_existing_add(&mut self, molecule: Molecule) {
        self.molecules.last().map(|mol| mol.id + 1).unwrap_or(1);
        self.molecules.push(molecule);
        self.molecule_cnt_list.push(0);
        self.n_molecules += 1;
    }

    /// Get the bonds of the current molecular system
    pub fn molsystem_bonds_get(&self) -> Option<(usize, Vec<i64>, Vec<i64>)> {
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut n_bonds = 0;
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            n_bonds +=
                usize::try_from(mol_count * mol.n_bonds).expect("amount of bonds to be positive");
        }

        if n_bonds == 0 {
            return None;
        }

        let mut from_atoms = Vec::with_capacity(n_bonds as usize);
        let mut to_atoms = Vec::with_capacity(n_bonds as usize);

        let atom_count = 0;
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            for _ in 0..*mol_count {
                for k in 0..mol.n_bonds {
                    let bond = mol.bonds[k as usize];

                    let from_atom = atom_count + bond.from_atom_id;
                    from_atoms.push(from_atom);

                    let to_atom = atom_count + bond.to_atom_id;
                    to_atoms.push(to_atom);
                }
            }
        }
        Some((n_bonds, from_atoms.clone(), to_atoms.clone()))
    }

    /// Translate from the particle numbering used in a frame set to the real
    /// particle numbering - used in the molecule description.
    fn particle_mapping_get_real_particle(
        &self,
        frame_set: &TrajectoryFrameSet,
        local: i64,
    ) -> Option<i64> {
        let n_blocks = frame_set.n_mapping_blocks;

        if n_blocks <= 0 {
            return Some(local);
        }

        for mapping in &frame_set.mappings {
            let first = mapping.num_first_particle;
            if local < first || local >= first + mapping.n_particles {
                continue;
            }
            return Some(
                mapping.real_particle_numbers
                    [usize::try_from(local - first).expect("local - first to usize")],
            );
        }

        None
    }

    /// Retrieve a vector (1D array) of particle data, from the last read frame set
    pub fn particle_data_vector(
        &mut self,
        is_particle_data: bool,
        block_id: BlockID,
    ) -> Option<(i64, Vec<f64>)> {
        let mut n_particles = 0;
        let mut block_index = -1;

        let data = if is_particle_data {
            self.particle_data_find(block_id)
        } else {
            self.data_find(block_id)
        };

        if data.is_none() {
            let mut block = GenBlock::new();
            let mut file_pos = self.get_file_position();

            // Read all blocks until next frame set block
            self.block_header_read(&mut block);
            loop {
                if file_pos >= self.input_file_len {
                    break;
                }
                match block.id {
                    BlockID::Unknown(0) | BlockID::TrajectoryFrameSet => break,
                    _ => {}
                }

                // Use hash by default (also TODO)
                self.block_read_next(&mut block);
                file_pos = self.get_file_position();
                if file_pos < self.input_file_len {
                    self.block_header_read(&mut block);
                }
            }

            let frame_set = &self.current_trajectory_frame_set;
            for i in 0..frame_set.n_particle_data_blocks {
                let data = &frame_set.tr_particle_data[i];
                if data.block_id == block_id {
                    block_index = i64::try_from(i).expect("index to i64");
                }
            }

            if block_index < 0 {
                return None;
            }
        }
        let data_unwrapped = data.expect("we just init");

        let frame_set = &self.current_trajectory_frame_set;
        if is_particle_data {
            let is_traj_block = self.current_trajectory_frame_set_input_file_pos > 0;

            n_particles = if is_traj_block && self.var_num_atoms {
                frame_set.n_particles
            } else {
                self.n_particles
            };
        }

        let data_type = data_unwrapped.data_type;

        let size = data_type.get_size();

        let n_frames = max(1, data_unwrapped.n_frames);
        let n_values_per_frame = data_unwrapped.n_values_per_frame;
        let stride_length = data_unwrapped.stride_length;

        let n_frames_div = (n_frames - 1) / stride_length + 1;
        let mut full_data_len =
            n_frames_div * i64::try_from(size).expect("size to i64") * n_values_per_frame;
        if is_particle_data {
            full_data_len *= n_particles;
        }

        let mut values = vec![0u8; full_data_len as usize];
        let unwrapped_values = data_unwrapped.values.expect("values to be avail");

        if !is_particle_data || frame_set.n_mapping_blocks <= 0 {
            values[..full_data_len as usize]
                .copy_from_slice(&unwrapped_values[..full_data_len as usize]);
        } else {
            let byte_per_particle = size * n_values_per_frame as usize;
            for i in 0..n_frames {
                for j in 0..n_particles {
                    let mapping = self
                        .particle_mapping_get_real_particle(frame_set, j)
                        .expect("from particle frame to real numbering");

                    let src_base = ((i * n_values_per_frame + j * n_values_per_frame)
                        * i64::try_from(size).expect("size to i64"))
                        as usize;
                    let dst_base = ((i * n_values_per_frame + mapping * n_values_per_frame)
                        * i64::try_from(size).expect("size to i64"))
                        as usize;

                    values[dst_base..dst_base + byte_per_particle]
                        .copy_from_slice(&unwrapped_values[src_base..src_base + byte_per_particle]);
                }
            }
        }

        let float_values: Vec<f64> = match data_type {
            DataType::Char => todo!("haven't implemented values to strings"),
            DataType::Int => values
                .chunks_exact(8)
                .map(|chunk| {
                    let arr = <[u8; 8]>::try_from(chunk).expect("Chunk should be 8 bytes");
                    i64::from_le_bytes(arr) as f64
                })
                .collect(),
            DataType::Float => values
                .chunks_exact(4)
                .map(|chunk| {
                    let arr = <[u8; 4]>::try_from(chunk).expect("Chunk should be 4 bytes");
                    f32::from_le_bytes(arr) as f64
                })
                .collect(),
            DataType::Double => values
                .chunks_exact(8)
                .map(|chunk| {
                    let arr = <[u8; 8]>::try_from(chunk).expect("Chunk should be 8 bytes");
                    f64::from_le_bytes(arr)
                })
                .collect(),
        };
        Some((n_particles, float_values))
    }

    /// Read one (the next) frame set, including particle mapping and related data blocks
    /// from the input_file of [`Self`]
    pub fn frame_set_read_next(&mut self) -> Result<(), ()> {
        self.input_file_init();

        let mut file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
        if file_pos < 0 && self.current_trajectory_frame_set_input_file_pos <= 0 {
            file_pos = self.first_trajectory_frame_set_input_pos;
        }

        if file_pos > 0 {
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("i64 to u64"),
                ))
                .expect("no error handling");
        } else {
            return Err(());
        }
        self.frame_set_read()
        // Ok(())
    }

    /// Read one frame set, including all particle mapping blocks and data blocks, starting from
    /// the current file position
    fn frame_set_read(&mut self) -> Result<(), ()> {
        self.input_file_init();
        let mut file_pos = self.get_file_position();
        let mut block = GenBlock::new();

        // Read block headers first to see what block is found
        self.block_header_read(&mut block);

        if block.id != BlockID::TrajectoryFrameSet || block.id == BlockID::Unknown(0) {
            return Err(());
        }

        self.current_trajectory_frame_set_input_file_pos =
            i64::try_from(file_pos).expect("u64 to i64");

        // TODO: make this fallible?
        self.block_read_next(&mut block);
        if block.id != BlockID::Unknown(0) {
            self.n_trajectory_frame_sets += 1;
            file_pos = self.get_file_position();

            // Read all blocks until next frame set block
            self.block_header_read(&mut block);
            loop {
                if file_pos >= self.input_file_len {
                    break;
                }
                match block.id {
                    BlockID::Unknown(0) | BlockID::TrajectoryFrameSet => break,
                    _ => {}
                }
                self.block_read_next(&mut block);
                file_pos = self.get_file_position();
                if file_pos < self.input_file_len {
                    self.block_header_read(&mut block);
                }
            }

            if block.id == BlockID::TrajectoryFrameSet {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(file_pos))
                    .expect("no error handling");
            }
        }
        Ok(())
    }

    /// Get the number of frame sets
    fn num_frame_sets_get(&mut self) -> i64 {
        let mut count = 0;
        let orig_frame_set = self.current_trajectory_frame_set.clone();
        let orig_frame_set_file_pos = self.current_trajectory_frame_set_input_file_pos;
        let mut file_pos = self.first_trajectory_frame_set_input_pos;

        if file_pos < 0 {
            self.n_trajectory_frame_sets = count;
            return count;
        }

        let mut block = GenBlock::new();
        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(
                u64::try_from(file_pos).expect("i64 to u64"),
            ))
            .expect("no error handling");
        self.current_trajectory_frame_set_input_file_pos = file_pos;

        // Read block headers first to see what block is found
        self.block_header_read(&mut block);

        if block.id != BlockID::TrajectoryFrameSet {
            panic!("cannot read block header at pos {file_pos}");
        }

        self.block_read_next(&mut block);
        count += 1;

        let long_stride_length = self.long_stride_length;
        let medium_stride_length = self.medium_stride_length;

        // Take long steps forward until a long step forward would be too long
        // or the last frame set is found
        file_pos = self
            .current_trajectory_frame_set
            .long_stride_next_frame_set_file_pos;

        while file_pos > 0 {
            if file_pos > 0 {
                count += long_stride_length;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }
            file_pos = self
                .current_trajectory_frame_set
                .long_stride_next_frame_set_file_pos;
        }

        // Take medium steps forward until a medium step forward would be too long or the
        // last frame set is found
        file_pos = self
            .current_trajectory_frame_set
            .medium_stride_next_frame_set_file_pos;
        while file_pos > 0 {
            if file_pos > 0 {
                count += medium_stride_length;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }
            file_pos = self
                .current_trajectory_frame_set
                .medium_stride_next_frame_set_file_pos;
        }

        // Take on step forward until the last frame set is found
        file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
        while file_pos > 0 {
            if file_pos > 0 {
                count += 1;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
        }

        self.n_trajectory_frame_sets = count;
        self.current_trajectory_frame_set = orig_frame_set;

        // from c code: The mapping block in the original frame set has been freed when reading
        // other frame sets
        self.current_trajectory_frame_set.mappings = Vec::new();
        self.current_trajectory_frame_set.n_mapping_blocks = 0;

        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(
                u64::try_from(self.first_trajectory_frame_set_input_pos).expect("i64 to u64"),
            ))
            .expect("no error handling");
        self.current_trajectory_frame_set_input_file_pos = orig_frame_set_file_pos;

        count
    }

    /// Find the requested frame set number
    pub fn frame_set_nr_find(&mut self, nr: i64) -> Result<(), ()> {
        let n_frame_sets = self.num_frame_sets_get();

        if nr >= n_frame_sets {
            return Err(());
        }

        let long_stride_length = self.long_stride_length;
        let medium_stride_length = self.medium_stride_length;
        let mut curr_nr = 0;

        // FIXME (from c): The frame set number of the current frame set is not stored

        let mut file_pos = if nr < n_frame_sets - 1 - nr {
            // Start from the beginning
            self.first_trajectory_frame_set_input_pos
        } else {
            // Start from the end
            curr_nr = n_frame_sets - 1;
            self.last_trajectory_frame_set_input_pos
        };

        if file_pos <= 0 {
            return Err(());
        }

        let mut block = GenBlock::new();
        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(
                u64::try_from(file_pos).expect("i64 to u64"),
            ))
            .expect("no error handling");
        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("cannot read block header at pos {file_pos}");
        }

        self.block_read_next(&mut block);

        if curr_nr == nr {
            return Ok(());
        }

        file_pos = self.current_trajectory_frame_set_input_file_pos;

        // Take long steps forward until a long step forward would be too long or
        // the right frame set is found
        while file_pos > 0 && curr_nr + long_stride_length <= nr {
            file_pos = self
                .current_trajectory_frame_set
                .long_stride_next_frame_set_file_pos;
            if file_pos > 0 {
                curr_nr += long_stride_length;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        // Take medium steps forward until a medium step forward would be too long or
        // the right frame set is found
        while file_pos > 0 && curr_nr + medium_stride_length <= nr {
            file_pos = self
                .current_trajectory_frame_set
                .medium_stride_next_frame_set_file_pos;
            if file_pos > 0 {
                curr_nr += medium_stride_length;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        // Take one step forward until the right frame set is found
        while file_pos > 0 && curr_nr < nr {
            file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
            if file_pos > 0 {
                curr_nr += 1;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        // Take long steps backward until a long step backward would be too long or
        // the right frame set is found
        while file_pos > 0 && curr_nr - long_stride_length >= nr {
            file_pos = self
                .current_trajectory_frame_set
                .long_stride_prev_frame_set_file_pos;
            if file_pos > 0 {
                curr_nr -= long_stride_length;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        // Take medium steps backward until a medium step backward would be too long or
        // the right frame set is found
        while file_pos > 0 && curr_nr - medium_stride_length >= nr {
            file_pos = self
                .current_trajectory_frame_set
                .medium_stride_prev_frame_set_file_pos;
            if file_pos > 0 {
                curr_nr -= medium_stride_length;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        // Take one step backward until the right frame set is found
        while file_pos > 0 && curr_nr > nr {
            file_pos = self.current_trajectory_frame_set.prev_frame_set_file_pos;
            if file_pos > 0 {
                curr_nr -= 1;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        // If for some reason the current frame set is not yet found
        // take one step forward until the right frame set is found
        while file_pos > 0 && curr_nr < nr {
            file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
            if file_pos > 0 {
                curr_nr += 1;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("i64 to u64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        Err(())
    }

    /// Read data from the current frame set from the `input_file`. Only read
    /// particle mapping and data blocks matching the specified [`BlockID`]
    pub fn frame_set_read_current_only_data_from_block_id(
        &mut self,
        match_block_id: BlockID,
    ) -> Result<(), ()> {
        let mut found_flag = false;
        self.input_file_init();

        let mut file_pos = self.current_trajectory_frame_set_input_file_pos;

        if file_pos < 0 {
            // No current frame set. This means that the first frame set must be read
            found_flag = true;
            file_pos = self.first_trajectory_frame_set_input_pos;
        }

        if file_pos > 0 {
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("i64 to u64"),
                ))
                .expect("no error handling");
        } else {
            return Err(());
        }

        let mut block = GenBlock::new();

        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("Cannot read block header at pos {file_pos}");
        }

        // If the current frame set had already been read skip its block destination
        if found_flag {
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("i64 to u64"),
                ))
                .expect("no error handling");
            // Otherwise read the frame set block
        } else {
            self.block_read_next(&mut block);
        }
        file_pos = i64::try_from(self.get_file_position()).expect("i64 from u64");

        found_flag = true;

        // Read only blocks of the request ID until next frame set block
        self.block_header_read(&mut block);
        while file_pos < i64::try_from(self.input_file_len).expect("i64 from u64")
            && block.id != BlockID::TrajectoryFrameSet
            && block.id != BlockID::Unknown(0)
        {
            if block.id == match_block_id {
                self.block_read_next(&mut block);
                file_pos = i64::try_from(self.get_file_position()).expect("i64 from u64");
                found_flag = true;
                if file_pos < i64::try_from(self.input_file_len).expect("i64 from u64") {
                    self.block_header_read(&mut block);
                }
            } else {
                file_pos += block.block_contents_size + block.header_contents_size;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Current(block.block_contents_size))
                    .expect("no error handling");
                if file_pos < i64::try_from(self.input_file_len).expect("i64 from u64") {
                    self.block_header_read(&mut block);
                }
            }
        }

        if block.id == BlockID::TrajectoryFrameSet {
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("u64 from i64"),
                ))
                .expect("no error handling");
        }

        if found_flag { Ok(()) } else { Err(()) }
    }

    /// Read one (the next) frame set, including particle mapping and data blocks with
    /// a specific block id from `input_file` of [`Self`]
    pub fn frame_set_read_next_only_data_from_block_id(
        &mut self,
        match_block_id: BlockID,
    ) -> Result<(), ()> {
        self.input_file_init();

        let mut file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
        if file_pos < 0 && self.current_trajectory_frame_set_input_file_pos <= 0 {
            file_pos = self.first_trajectory_frame_set_input_pos;
        }

        if file_pos > 0 {
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("u64 from i64"),
                ))
                .expect("no error handling");
        } else {
            return Err(());
        }

        let mut block = GenBlock::new();

        // Read block headers first to see what block is found
        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("Cannot read block header at pos {file_pos}");
        }

        self.current_trajectory_frame_set_input_file_pos = file_pos;

        self.block_read_next(&mut block);

        self.frame_set_read_current_only_data_from_block_id(match_block_id)
    }

    /// Get the name of a data block of a specific ID.
    pub fn data_block_name_get(&mut self, match_block_id: BlockID) -> Result<String, ()> {
        for i in 0..self.n_particle_data_blocks {
            let data = &self.non_tr_particle_data[i];
            if data.block_id == match_block_id {
                return Ok(data.block_name.clone());
            }
        }

        for i in 0..self.n_data_blocks {
            let data = &self.non_tr_data[i];
            if data.block_id == match_block_id {
                return Ok(data.block_name.clone());
            }
        }

        let particle_data_result = self.particle_data_find(match_block_id);
        let mut particle_block_data = false;
        if particle_data_result.is_some() {
            particle_block_data = true;
        } else {
            let data_result = self.data_find(match_block_id);
            if data_result.is_some() {
                particle_block_data = false;
            } else {
                let result = self.frame_set_read_current_only_data_from_block_id(match_block_id);
                if result.is_err() {
                    return Err(());
                }
                let particle_data_result = self.particle_data_find(match_block_id);
                if particle_data_result.is_some() {
                    particle_block_data = true;
                } else {
                    let data_result = self.data_find(match_block_id);
                    if data_result.is_some() {
                        particle_block_data = false;
                    }
                }
            }
        }

        let frame_set = &self.current_trajectory_frame_set;
        if particle_block_data {
            for i in 0..frame_set.n_particle_data_blocks {
                let data = &frame_set.tr_particle_data[i];
                if data.block_id == match_block_id {
                    return Ok(data.block_name.clone());
                }
            }
        } else {
            for i in 0..frame_set.n_data_blocks {
                let data = &frame_set.tr_data[i];
                if data.block_id == match_block_id {
                    return Ok(data.block_name.clone());
                }
            }
        }

        Err(())
    }
}
