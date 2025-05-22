use log::warn;

use crate::MAX_STR_LEN;
use crate::data::Data;
use crate::gen_block::{BlockID, GenBlock};
use crate::molecule::Molecule;
use crate::trajectory_frame_set::TrajectoryFrameSet;
use core::panic;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

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
    pub molecule_counts: Vec<i64>,
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
            molecule_counts: Vec::new(),
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
            &path.to_str().unwrap()[..(MAX_STR_LEN - 1)]
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

    fn read_i64_le_bytes(&mut self) -> i64 {
        let mut buf = [0u8; 8];
        self.input_file
            .as_mut()
            .expect("input_file should be init")
            .read_exact(&mut buf)
            .expect("we dont handle errors yet");
        i64::from_le_bytes(buf)
    }

    fn read_u64_le_bytes(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.input_file
            .as_mut()
            .expect("input_file should be init")
            .read_exact(&mut buf)
            .expect("we dont handle errors yet");
        u64::from_le_bytes(buf)
    }

    fn read_bool_le_bytes(&mut self) -> bool {
        let mut buf = [0u8; 1];
        self.input_file
            .as_mut()
            .expect("input_file should be init")
            .read_exact(&mut buf)
            .expect("we dont handle errors yet");
        buf[0] != 0
    }

    fn read_md5_hash(&mut self) -> i64 {
        let mut buf = [0u8; 8];
        self.input_file
            .as_mut()
            .expect("input_file should be init")
            .read_exact(&mut buf)
            .expect("we dont handle errors yet");
        i64::from_le_bytes(buf)
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

        block.header_contents_size = self.read_i64_le_bytes();
        dbg!(&block.header_contents_size);

        if block.header_contents_size == 0 {
            block.id = BlockID::Undetermined;
            warn!("header_contents_size was 0 block.id is Undetermined");
        }

        block.block_contents_size = self.read_i64_le_bytes();
        dbg!(&block.block_contents_size);

        block.id = BlockID::from_i64(self.read_i64_le_bytes());
        dbg!(&block.id);

        self.input_file
            .as_mut()
            .expect("we just init file")
            .read_exact(&mut block.md5_hash)
            .expect("no error handling");
        dbg!(block.md5_hash);

        block.name = Some(self.fread_str());
        dbg!(&block.name);

        block.version = self.read_u64_le_bytes();
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

    fn frame_set_block_read(&mut self, block: &mut GenBlock) {}
    fn trajectory_mapping_block_read(&mut self, block: &mut GenBlock) {}
    fn general_info_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();

        let start_pos = self
            .input_file
            .as_mut()
            .expect("we just init input_file")
            .stream_position()
            .expect("no error handling");

        self.first_program_name = self.fread_str();
        dbg!(&self.first_program_name);
        self.last_program_name = self.fread_str();
        dbg!(&self.last_program_name);
        self.first_user_name = self.fread_str();
        dbg!(&self.first_user_name);
        self.last_user_name = self.fread_str();
        dbg!(&self.last_user_name);
        self.first_computer_name = self.fread_str();
        dbg!(&self.first_computer_name);
        self.last_computer_name = self.fread_str();
        dbg!(&self.last_computer_name);
        self.first_pgp_signature = self.fread_str();
        dbg!(&self.first_pgp_signature);
        self.last_pgp_signature = self.fread_str();
        dbg!(&self.last_pgp_signature);
        self.forcefield_name = self.fread_str();
        dbg!(&self.forcefield_name);

        self.creation_time = self.read_i64_le_bytes();
        dbg!(&self.creation_time);
        self.var_num_atoms = self.read_bool_le_bytes();
        dbg!(&self.var_num_atoms);
        self.frame_set_n_frames = self.read_i64_le_bytes();
        dbg!(&self.frame_set_n_frames);
        self.first_trajectory_frame_set_input_pos = self.read_i64_le_bytes();
        dbg!(&self.first_trajectory_frame_set_input_pos);

        self.current_trajectory_frame_set.next_frame_set_file_pos =
            self.first_trajectory_frame_set_input_pos;
        self.last_trajectory_frame_set_input_pos = self.read_i64_le_bytes();
        dbg!(&self.last_trajectory_frame_set_input_pos);

        self.medium_stride_length = self.read_i64_le_bytes();
        dbg!(&self.medium_stride_length);

        self.long_stride_length = self.read_i64_le_bytes();
        dbg!(&self.long_stride_length);

        if block.version >= 3 {
            self.distance_unit_exponential = self.read_i64_le_bytes();
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
    fn molecules_block_read(&mut self, block: &mut GenBlock) {}

    fn block_read_next(&mut self, block: &mut GenBlock) {
        match block.id {
            BlockID::TrajectoryFrameSet => self.frame_set_block_read(block),
            BlockID::ParticleMapping => self.trajectory_mapping_block_read(block),
            BlockID::GeneralInfo => self.general_info_block_read(block),
            BlockID::Molecules => self.molecules_block_read(block),
            BlockID::Undetermined => todo!("undetermined arm of block_read_next"),
        }
    }

    pub fn fread_str(&mut self) -> String {
        // Accumulate bytes here, including the trailing 0:
        let mut buf: Vec<u8> = Vec::new();

        // Temporary single-byte buffer for reading:
        let mut byte = [0u8; 1];

        loop {
            // Try to read exactly one byte
            match self
                .input_file
                .as_mut()
                .expect("init input_file")
                .read_exact(&mut byte)
            {
                Ok(()) => {
                    let b = byte[0];
                    buf.push(b);

                    // // If hash_mode, feed it into the MD5 state right away
                    // if hash_mode {
                    //     md5_state.update(&[b]);
                    // }

                    // Stop on NUL or if we exceed max length
                    if b == 0 {
                        break;
                    }
                    if buf.len() >= MAX_STR_LEN {
                        // We hit the same “limit” as the C version (avoid overflow).
                        // In C, it would stop copying; here we break and use what we have.
                        break;
                    }
                }
                Err(_) => {
                    panic!("something went wrong when reading string");
                    // // EOF or other I/O error. Distinguish EOF (Minor) vs. other errors (Critical).
                    // if e.kind() == io::ErrorKind::UnexpectedEof {
                    //     // Clear the EOF flag (if needed) and return “minor” failure.
                    //     return Err(ReadStrError::Minor);
                    // } else {
                    //     return Err(ReadStrError::Critical(e));
                    // }
                }
            }
        }

        // If we read exactly one byte and it was NUL, that's effectively an empty string.
        // The C version would then allocate a length-1 buffer and store "\0".
        // We’ll treat that as an empty Rust String here (dropping the trailing zero):
        if buf.len() == 1 && buf[0] == 0 {
            return String::new();
        }

        // Otherwise, we have some bytes like [b'a', b'b', b'c', 0]. Drop the final 0:
        if let Some(&0) = buf.last() {
            buf.pop();
        }

        // Convert to UTF-8, with a lossy fallback if it wasn’t valid UTF-8:
        match String::from_utf8(buf) {
            Ok(valid) => valid,
            Err(e) => String::from_utf8_lossy(&e.into_bytes()).into_owned(),
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

                // 3) If `block.id == -1` or `block.id == TNG_TRAJECTORY_FRAME_SET`, exit loop
                match block.id {
                    BlockID::Undetermined | BlockID::TrajectoryFrameSet => break,
                    _ => {}
                }

                self.block_read_next(&mut block);
            }
        }
    }
}
