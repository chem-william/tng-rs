use log::{debug, error, warn};
use std::cmp::{max, min};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::TngError;

use crate::atom::Atom;
use crate::bond::Bond;
use crate::chain::Chain;
use crate::compress::{
    tng_compress_pos, tng_compress_pos_float, tng_compress_vel, tng_compress_vel_float,
};
use crate::data::{Compression, Data, DataType};
use crate::gen_block::{BlockID, GenBlock};
use crate::molecule::Molecule;
use crate::particle_mapping::ParticleMapping;
use crate::residue::Residue;
use crate::trajectory_frame_set::TrajectoryFrameSet;
use crate::utils::{Endianness32, Endianness64, SwapFn32, SwapFn64, bounded_len};
use crate::{FRAME_DEPENDENT, MAX_STR_LEN, PARTICLE_DEPENDENT, utils};
use core::panic;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

const USE_HASH: bool = false;

fn is_same_file(file1: &File, file2: &File) -> std::io::Result<bool> {
    let meta1 = file1.metadata()?;
    let meta2 = file2.metadata()?;

    Ok(meta1.ino() == meta2.ino() && meta1.dev() == meta2.dev())
}

/// This returns the number of integers required for the storage of the algorithm with
/// the best compression ratio
fn tng_compress_nalgo() -> u64 {
    // There are currently four parameters required
    // 1) The compression algorithm for the first frame (initial_coding).
    // 2) One parameter to the algorithm for the first frame (the initial coding parameter).
    // 3) The compression algorithm for the remaining frames (coding).
    // 4) One parameter to the algorithm for the remaining frames (the coding parameter).
    4
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

    /// The endianness of 32 bit values of the current computer
    pub endianness32: Endianness32,
    /// The endianness of 64 bit values of the current computer
    pub endianness64: Endianness64,
    /// Closure to swap 32 bit values to and from the endianness of the input file
    pub input_swap32: Option<SwapFn32>,
    /// Closure to swap 64 bit values to and from the endianness of the input file
    pub input_swap64: Option<SwapFn64>,
    /// Closure to swap 32 bit values to and from the endianness of the input file
    pub output_swap32: Option<SwapFn32>,
    /// Closure to swap 64 bit values to and from the endianness of the output file
    pub output_swap64: Option<SwapFn64>,

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
    pub time: u64,
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

#[derive(Copy, Clone, Debug)]
enum Slot {
    NonTr,
    NonTrParticle,
    Tr,
    TrParticle,
}

impl Trajectory {
    /// Detect the host’s native 32‐ and 64‐bit endianness and store the corresponding enum values.
    ///
    /// # Panic
    /// Panics if unable to detect either the 32- or 64-bit endianness
    pub fn detect_host_endianness() -> (Endianness32, Endianness64) {
        let probe32: i32 = 0x01234567;
        let first_byte32 = probe32.to_ne_bytes()[0];
        let endianness32 = match first_byte32 {
            0x01 => Endianness32::Big,
            0x67 => Endianness32::Little,
            0x45 => Endianness32::BytePairSwap,
            _ => panic!("unable to detect host system 32-bit endianness"),
        };

        let probe64: i64 = 0x0123_4567_89AB_CDEF;
        let first_byte64 = probe64.to_ne_bytes()[0];
        let endianness64 = match first_byte64 {
            0x01 => Endianness64::Big,
            0xEF => Endianness64::Little,
            0x89 => Endianness64::QuadSwap,
            0x45 => Endianness64::BytePairSwap,
            0x23 => Endianness64::ByteSwap,
            _ => panic!("unable to detect host system 64-bit endianness"),
        };

        debug!("Host system 32-bit endianness: {endianness32:?}");
        debug!("Host system 64-bit endianness: {endianness64:?}");

        (endianness32, endianness64)
    }

    pub fn new() -> Self {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("able to get time since UNIX_EPOCH")
            .as_secs();

        let (endianness32, endianness64) = Trajectory::detect_host_endianness();

        Trajectory {
            input_file_path: PathBuf::new(),
            input_file: None,
            input_file_len: 0,

            output_file_path: PathBuf::new(),
            output_file: None,
            endianness32,
            endianness64,
            input_swap32: None,
            input_swap64: None,
            output_swap32: None,
            output_swap64: None,

            first_program_name: String::new(),
            forcefield_name: String::new(),
            first_user_name: String::new(),
            first_computer_name: String::new(),
            first_pgp_signature: String::new(),

            last_program_name: String::new(),
            last_user_name: String::new(),
            last_computer_name: String::new(),
            last_pgp_signature: String::new(),

            time,
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

    fn get_input_file_position(&self) -> u64 {
        self.input_file
            .as_ref()
            .expect("init input_file")
            .stream_position()
            .expect("no error handling")
    }

    fn get_output_file_position(&self) -> u64 {
        self.output_file
            .as_ref()
            .expect("init output_file")
            .stream_position()
            .expect("no error handling")
    }

    // c function: tng_output_file_set
    /// Set the name of the output file.
    pub fn set_output_file(&mut self, path: &Path) {
        if self.output_file_path == path {
            return;
        }

        // If a file was already open, drop (close) it.
        self.output_file.take();

        let truncated = if path.to_str().expect("valid unicode path").len() + 1 > MAX_STR_LEN {
            &path.to_str().unwrap()[..MAX_STR_LEN - 1]
        } else {
            path.to_str().unwrap()
        };

        self.output_file_path = PathBuf::from(truncated.to_string());

        self.output_file_init();
    }

    /// Open the output file is it is not already opened. If the file does not
    /// already exist, create it.
    pub fn output_file_init(&mut self) {
        if self.output_file.is_none() {
            // If no path has ever been set, error out
            if self.output_file_path.as_os_str().is_empty() {
                eprintln!("No file specified for reading. {}:{}", file!(), line!());
                panic!();
            }

            // Try to create the file
            let path = self.output_file_path.clone();
            match File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
            {
                Ok(f) => {
                    self.output_file = Some(f);
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
    }

    pub fn set_medium_stride_length(&mut self, length: i64) -> Result<(), TngError> {
        if length >= self.long_stride_length {
            return Err(TngError::Constraint(format!(
                "medium stride length ({length}) must be less than long stride length ({})",
                self.long_stride_length
            )));
        }
        self.medium_stride_length = length;

        Ok(())
    }

    pub fn set_long_stride_length(&mut self, length: i64) -> Result<(), TngError> {
        if length <= self.medium_stride_length {
            return Err(TngError::Constraint(format!(
                "long stride length ({length}) must be greater than medium stride length ({})",
                self.medium_stride_length
            )));
        }
        self.long_stride_length = length;

        Ok(())
    }

    pub fn set_first_user_name(&mut self, new_name: &str) {
        // We use `floor_char_boundary` as Rust strings has to be valid UTF-8. This way we never split a charachter
        // and never panic on non-UTF-8 names
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.first_user_name = new_name[..length].to_string();
    }

    pub fn set_first_computer_name(&mut self, new_name: &str) {
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.first_computer_name = new_name[..length].to_string();
    }

    pub fn set_first_program_name(&mut self, new_name: &str) {
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.first_program_name = new_name[..length].to_string();
    }

    pub fn set_forcefield_name(&mut self, new_name: &str) {
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.forcefield_name = new_name[..length].to_string();
    }

    pub fn set_time_per_frame(&mut self, time: f64) -> Result<(), TngError> {
        if time < 0.0 {
            return Err(TngError::Constraint(format!(
                "The time per frame must be >= 0. Currently is {time}"
            )));
        }

        if (time - self.time_per_frame).abs() < 0.00001 {
            return Ok(());
        }

        // If the current frame set is not finished write it to disk before changing per frame
        if self.time_per_frame > 0.0 && self.current_trajectory_frame_set.n_unwritten_frames > 0 {
            self.current_trajectory_frame_set.n_frames =
                self.current_trajectory_frame_set.n_unwritten_frames;
            self.frame_set_write(USE_HASH)?;
        }
        self.time_per_frame = time;
        Ok(())
    }

    pub fn get_num_particles(&self) -> i64 {
        if self.var_num_atoms {
            self.n_particles
        } else {
            self.current_trajectory_frame_set.n_particles
        }
    }

    // c function: tng_input_file_set
    /// Set the name of the input file.
    pub fn set_input_file(&mut self, path: &Path) {
        if self.input_file_path == path {
            return;
        }

        // If a file was already open, drop (close) it.
        self.input_file.take();

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

    /// Open the input file if it is not already opened.
    pub fn input_file_init(&mut self) {
        if self.input_file.is_none() {
            // If no path has been set, error out
            if self.input_file_path.as_os_str().is_empty() {
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
                        "Cannot open file {}. {}:{}",
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
    fn block_header_read(&mut self, block: &mut GenBlock) -> Result<(), TngError> {
        self.input_file_init();

        let start_pos = self.get_input_file_position();

        let header_contents_size_bytes =
            utils::read_exact_array::<8, _>(self.input_file.as_mut().expect("init input_file"));
        block.header_contents_size = u64::from_ne_bytes(header_contents_size_bytes);

        if block.header_contents_size == 0 {
            block.id = BlockID::Unknown;
            return Err(TngError::Constraint(
                "header_contents_size was 0. block.id is BlockID::Unknown".to_string(),
            ));
        }

        // If this was the size of the general info block, check the endianness
        if self.get_input_file_position() < 9 {
            // File is little endian
            if header_contents_size_bytes[0] != 0 && header_contents_size_bytes[7] == 0 {
                // If the architecture endianess is little endian no byte swap will be needed.
                // Otherwise use the functions to swap to little endian
                if self.endianness32 == Endianness32::Little {
                    self.input_swap32 = None;
                } else {
                    self.input_swap32 = Some(utils::swap_byte_order_little_endian_32);
                }

                if self.endianness64 == Endianness64::Little {
                    self.input_swap64 = None;
                } else {
                    self.input_swap64 = Some(utils::swap_byte_order_little_endian_64);
                }
            }
            // File is big endian
            else {
                // If the architecture endianness is big endian no byte swap
                // will be needed. Otherwise use the functions to swap to big endian
                if self.endianness32 == Endianness32::Big {
                    self.input_swap32 = None;
                } else {
                    self.input_swap32 = Some(utils::swap_byte_order_little_endian_32);
                }

                if self.endianness64 == Endianness64::Big {
                    self.input_swap64 = None;
                } else {
                    self.input_swap64 = Some(utils::swap_byte_order_big_endian_64);
                }
            }
        }

        if let Some(swap_fn) = self.input_swap64 {
            swap_fn(self.endianness64, &mut block.header_contents_size);
        }

        let inp_file = self.input_file.as_mut().expect("init input_file");

        block.block_contents_size = utils::read_u64(inp_file, self.endianness64, self.input_swap64);

        block.id = BlockID::from_u64(utils::read_u64(
            inp_file,
            self.endianness64,
            self.input_swap64,
        ));

        inp_file
            .read_exact(&mut block.md5_hash)
            .expect("no error handling");

        block.name = Some(utils::fread_str(inp_file));

        block.version = utils::read_u64(inp_file, self.endianness64, self.input_swap64);

        let new_pos = (start_pos as i128 + block.header_contents_size as i128)
            .try_into()
            .expect("set new position when reading block header");

        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            // TODO
            .expect("no error handling");
        Ok(())
    }

    fn frame_set_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();
        let start_pos = self.get_input_file_position();

        // FIXME (from c): Does not check if the size of the contents matches the
        // expected size of if the contents can be read
        let file_pos = start_pos - block.header_contents_size;
        self.current_trajectory_frame_set_input_file_pos =
            i64::try_from(file_pos).expect("u64 to i64");
        // set_particle_mapping_free
        let frame_set = &mut self.current_trajectory_frame_set;
        let inp_file = self.input_file.as_mut().expect("init input_file");
        frame_set.first_frame = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        frame_set.n_frames = utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        if self.var_num_atoms {
            // let prev_n_particles = frame_set.n_particles;
            frame_set.n_particles = 0;

            for (mol, mol_count) in self
                .molecules
                .iter()
                .zip(frame_set.molecule_cnt_list.iter_mut())
            {
                *mol_count = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
                frame_set.n_particles += mol.n_atoms * *mol_count;
            }

            // from the c code
            // if prev_n_particles && frame_set.n_particles != prev_n_particles {
            //     /* FIXME: Particle dependent data memory management */
            // }
        }

        frame_set.next_frame_set_file_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        frame_set.prev_frame_set_file_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        frame_set.medium_stride_next_frame_set_file_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        frame_set.medium_stride_prev_frame_set_file_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        frame_set.long_stride_next_frame_set_file_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        frame_set.long_stride_prev_frame_set_file_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        if block.version >= 3 {
            frame_set.first_frame_time =
                utils::read_f64(inp_file, self.endianness64, self.input_swap64);
            self.time_per_frame = utils::read_f64(inp_file, self.endianness64, self.input_swap64);
        } else {
            frame_set.first_frame_time = -1.0;
            self.time_per_frame = -1.0;
        }

        // TODO: Hash mode
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(start_pos + block.block_contents_size))
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
        self.input_file_init();

        let start_pos = self.get_input_file_position();
        let inp_file = self.input_file.as_mut().expect("init input_file");

        // FIXME (from c): Does not check if the size of the contents matches the
        // expected size of if the contents can be read
        let frame_set = &mut self.current_trajectory_frame_set;
        frame_set.n_mapping_blocks += 1;
        let mut mapping = ParticleMapping::new();

        // TODO: hash mode

        mapping.num_first_particle =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        mapping.n_particles = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
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
            .seek(SeekFrom::Start(start_pos + block.block_contents_size))
            .expect("no error handling");
    }

    /// WRite the atom mappings of the current trajectory frame set
    /// `block` is a general block container
    /// `mapping_block_nr` is the index of the mapping block to write
    /// `hash_mode` is an option to decide whether to use the md5 hash or not
    /// if hash_mode == USE_HASH an md5 hash will be generated and written
    fn trajectory_mapping_block_write(
        &mut self,
        block: &mut GenBlock,
        mapping_block_nr: usize,
        _hash_mode: bool,
    ) -> Result<(), TngError> {
        self.output_file_init();

        block.name = Some("PARTICLE MAPPING".to_string());
        block.id = BlockID::ParticleMapping;

        let mapping = &self.current_trajectory_frame_set.mappings[mapping_block_nr];
        block.block_contents_size =
            u64::try_from(self.trajectory_mapping_block_len_calculate(mapping.n_particles))
                .expect("u64 from usize");

        // TODO: hash mode
        let header_file_pos = self
            .output_file
            .as_mut()
            .expect("init output_file")
            .stream_position()
            // TODO
            .expect("no error handling");
        self.block_header_write(block);

        // TODO: hash mode

        let out_file = self.output_file.as_mut().expect("init output_file");
        let mapping = &self.current_trajectory_frame_set.mappings[mapping_block_nr];
        utils::write_i64(
            out_file,
            mapping.num_first_particle,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_i64(
            out_file,
            mapping.n_particles,
            self.endianness64,
            self.output_swap64,
        );

        // we don't need the if-else branch here from the C code as utils::write_* already handles the
        // case where output_swap64 is None
        // line 3946 tng_io.c
        // TODO: hash mode
        for i in 0..mapping.n_particles {
            utils::write_i64(
                out_file,
                mapping.real_particle_numbers[usize::try_from(i).expect("usize from i64")],
                self.endianness64,
                self.output_swap64,
            );
        }

        // TODO: hash mode
        // lien 3973 tng_io.c

        Ok(())
    }

    fn general_info_block_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();

        let start_pos = self.get_input_file_position();
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

        self.time = utils::read_u64(inp_file, self.endianness64, self.input_swap64);
        self.var_num_atoms = utils::read_bool_le_bytes(inp_file);
        self.frame_set_n_frames = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        self.first_trajectory_frame_set_input_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        self.current_trajectory_frame_set.next_frame_set_file_pos =
            self.first_trajectory_frame_set_input_pos;
        self.last_trajectory_frame_set_input_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        self.medium_stride_length = utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        self.long_stride_length = utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        if block.version >= 3 {
            self.distance_unit_exponential =
                utils::read_i64(inp_file, self.endianness64, self.input_swap64);
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
        let start_pos = self.get_input_file_position();

        self.molecules.clear();

        self.n_molecules = utils::read_i64(
            self.input_file.as_mut().expect("init input_file"),
            self.endianness64,
            self.input_swap64,
        );

        self.n_particles = 0;
        self.molecules = Vec::with_capacity(self.n_molecules as usize);

        if !self.var_num_atoms {
            self.molecule_cnt_list = Vec::with_capacity(self.n_molecules as usize);
        }

        // Read each molecule from file
        for mol_idx in 0..self.n_molecules {
            let inp_file = self.input_file.as_mut().expect("init input_file");
            let mut molecule = Molecule::new();
            molecule.id = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
            molecule.name = utils::fread_str(inp_file);
            molecule.quaternary_str =
                utils::read_i64(inp_file, self.endianness64, self.input_swap64);

            if !self.var_num_atoms {
                let count = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
                self.molecule_cnt_list.push(count);
            }

            molecule.n_chains = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
            molecule.n_residues = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
            molecule.n_atoms = utils::read_i64(inp_file, self.endianness64, self.input_swap64);

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
                let mut chain = Chain::new();

                // Link back to parent molecule index
                chain.parent_molecule_idx = mol_idx as usize;
                chain.name = String::new();

                chain.read_data(self);

                // Determine this chain’s slice of `self.residues`:
                let start = residue_idx;
                let end = start + chain.n_residues;
                chain.residues_indices = (start as usize, end as usize);
                residue_idx = end; // next free residue slot

                // Read the residues of the chain
                for local_idx in start..end {
                    // let residue = &mut molecule.residues[local_idx as usize];
                    let mut residue = Residue::new();

                    // Link back to parent chain index
                    residue.chain_index = Some(chain_idx as usize);
                    residue.name = String::new();

                    residue.read_data(self);

                    // Compute atoms_offset = `atom - molecule->atoms` in C
                    residue.atoms_offset = atom_idx;

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
            molecule.n_bonds = utils::read_i64(inp_file, self.endianness64, self.input_swap64);

            for _ in 0..molecule.n_bonds {
                let mut bond = Bond::new();
                bond.from_atom_id = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
                bond.from_atom_id = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
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
        block_meta_info.n_values = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        block_meta_info.codec_id = Compression::from_i64(utils::read_i64(
            inp_file,
            self.endianness64,
            self.input_swap64,
        ));

        block_meta_info.multiplier = if block_meta_info.codec_id != Compression::Uncompressed {
            utils::read_f64(inp_file, self.endianness64, self.input_swap64)
        } else {
            1.0
        };

        if block_meta_info.dependency & FRAME_DEPENDENT != 0 {
            if block_meta_info.sparse_data != 0 {
                block_meta_info.first_frame_with_data =
                    utils::read_i64(inp_file, self.endianness64, self.input_swap64);
                block_meta_info.stride_length =
                    utils::read_i64(inp_file, self.endianness64, self.input_swap64);
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
            block_meta_info.num_first_particle =
                utils::read_i64(inp_file, self.endianness64, self.input_swap64);
            block_meta_info.block_n_particles =
                utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        } else {
            block_meta_info.num_first_particle = -1;
            block_meta_info.block_n_particles = 0;
        }

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

    // We don't bother to estimate the max bound of the compression as Rust has dynamic arrays (Vec)
    // which C doesn't thus needing to pre-allocate the maximum amount.
    fn gzip_compress(data: &[u8], len: usize) -> Result<usize, ()> {
        let mut encoder = ZlibEncoder::new(Vec::with_capacity(len), flate2::Compression::default());
        encoder.write_all(data).map_err(|_| ())?;

        let compressed = encoder.finish().map_err(|_| ())?;
        Ok(compressed.len())
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
            if is_particle_data {
                self.particle_data_block_create(is_traj_block);
            } else {
                self.data_block_create(is_traj_block);
            }

            let frame_set = &mut self.current_trajectory_frame_set;
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
        let (actual_contents, full_data_len) = if data.codec_id != Compression::Uncompressed {
            let mut full_data_len = (n_frames_div as usize)
                .checked_mul(size)
                .and_then(|x| x.checked_mul(meta_info.n_values as usize))
                .unwrap_or(0);
            if is_particle_data {
                full_data_len = full_data_len
                    .checked_mul(meta_info.block_n_particles as usize)
                    .expect("mul of meta_info.block_n_particles");
            }

            let actual_contents = match data.codec_id {
                Compression::Uncompressed => unreachable!(),
                Compression::XTC => todo!("XTC compression not implemented yet"),
                Compression::TNG => todo!("TNG is todo"),
                Compression::GZip => {
                    Trajectory::gzip_uncompress(&contents, block_data_len, full_data_len)?
                }
            };
            (actual_contents, full_data_len)
        } else {
            let full_data_len = block_data_len as usize;
            (contents, full_data_len)
        };

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

            // Endianness is handled by the TNG compression library. TNG compressed blocks are always written as little endian by the compression library
            if data.codec_id != Compression::TNG {
                match data.data_type {
                    DataType::Float => {
                        if let Some(input_swap32) = self.input_swap32 {
                            for chunk in data
                                .values
                                .as_mut()
                                .expect("data to be some")
                                .chunks_exact_mut(size)
                            {
                                let mut val = u32::from_ne_bytes(chunk.try_into().unwrap());
                                input_swap32(self.endianness32, &mut val);

                                chunk.copy_from_slice(&val.to_ne_bytes());
                            }
                        }
                    }
                    DataType::Int | DataType::Double => {
                        if let Some(input_swap64) = self.input_swap64 {
                            for chunk in data
                                .values
                                .as_mut()
                                .expect("data to be some")
                                .chunks_exact_mut(size)
                            {
                                let mut val = u64::from_ne_bytes(chunk.try_into().unwrap());
                                input_swap64(self.endianness64, &mut val);

                                chunk.copy_from_slice(&val.to_ne_bytes());
                            }
                        }
                    }
                    DataType::Char => {}
                }
            }
        }
        Ok(())
    }

    /// Read the contents of a data block (particle or non-particle data)
    fn data_block_contents_read(&mut self, block: &mut GenBlock) {
        self.input_file_init();
        let start_pos = self.get_input_file_position();

        let meta_info = self.data_block_meta_information_read(block);

        let current_pos = self.get_input_file_position();
        let remaining_len = block.block_contents_size - (current_pos - start_pos);

        self.data_read(block, meta_info, remaining_len);

        // TODO: handle md5 hash

        // if hash_mode == TNG_USE_HASH {}

        let new_pos = start_pos + block.block_contents_size;
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
                    let current_pos = self.get_input_file_position();
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
            self.block_header_read(&mut block);
            while prev_pos < self.input_file_len
                && block.id != BlockID::Unknown
                && block.id != BlockID::TrajectoryFrameSet
            {
                println!("calling block_read_next");
                self.block_read_next(&mut block);
                prev_pos = self.get_input_file_position();
                self.block_header_read(&mut block);
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

    /// Calculate the total byte length of the “general info” block header.
    fn general_info_block_len_calculate(&self) -> u64 {
        // In C, each `char*` must be non-NULL. In Rust, `String` is never null.
        // It's also guaranteed to be at least empty if we have a Trajectory
        // due to Trajectory::new()
        // so we just don't do anything

        let first_program_name_len = bounded_len(&self.first_program_name);
        let last_program_name_len = bounded_len(&self.last_program_name);
        let first_user_name_len = bounded_len(&self.first_user_name);
        let last_user_name_len = bounded_len(&self.last_user_name);
        let first_computer_name_len = bounded_len(&self.first_computer_name);
        let last_computer_name_len = bounded_len(&self.last_computer_name);
        let first_pgp_signature_len = bounded_len(&self.first_pgp_signature);
        let last_pgp_signature_len = bounded_len(&self.last_pgp_signature);
        let forcefield_name_len = bounded_len(&self.forcefield_name);

        // Sum fixed‐size numeric fields:
        let mut total: usize = 0;
        total += size_of::<u64>(); // time
        total += size_of::<u8>(); // var_num_atoms
        total += size_of::<i64>(); // frame_set_n_frames
        total += size_of::<i64>(); // first_trajectory_frame_set_input_file_pos
        total += size_of::<i64>(); // last_trajectory_frame_set_input_file_pos
        total += size_of::<i64>(); // medium_stride_length
        total += size_of::<i64>(); // long_stride_length
        total += size_of::<i64>(); // distance_unit_exponential (assume C’s int is 32 bits)

        // Add all string lengths:
        total += first_program_name_len;
        total += last_program_name_len;
        total += first_user_name_len;
        total += last_user_name_len;
        total += first_computer_name_len;
        total += last_computer_name_len;
        total += first_pgp_signature_len;
        total += last_pgp_signature_len;
        total += forcefield_name_len;

        u64::try_from(total).expect("u64 from usize")
    }

    fn molecules_block_len_calculate(&self) -> u64 {
        let mut length = 0;
        for molecule in &self.molecules {
            length += bounded_len(&molecule.name);
            for chain in &molecule.chains {
                length += size_of::<u64>(); // chain.id
                length += bounded_len(&chain.name);
                length += size_of::<u64>(); // chain.n_residues
            }

            for residue in &molecule.residues {
                length += size_of::<u64>(); // residue.id
                length += bounded_len(&residue.name);
                length += size_of::<u64>(); // residue.n_atoms
            }

            for atom in &molecule.atoms {
                length += size_of::<i64>(); // atom.id
                length += bounded_len(&atom.name);
                length += bounded_len(&atom.atom_type);
            }

            for _ in &molecule.bonds {
                length += size_of::<i64>() + size_of::<i64>(); // bond.from_atom_id + bond.to_atom_id
            }
        }

        let proto_mol = &self.molecules[0];
        length += size_of_val(&self.n_molecules);
        length += size_of_val(&proto_mol.id);
        length += size_of_val(&proto_mol.quaternary_str);
        length += size_of_val(&proto_mol.n_chains);
        length += size_of_val(&proto_mol.n_residues);
        length += size_of_val(&proto_mol.n_atoms);
        length += size_of_val(&proto_mol.n_bonds);
        length += size_of_val(&self.n_molecules);

        if !self.var_num_atoms {
            length += usize::try_from(self.n_molecules).expect("usize from i64") * size_of::<u64>();
        }
        u64::try_from(length).expect("u64 from usize")
    }

    pub fn data_block_len_calculate(
        data: &Data,
        is_particle_data: bool,
        n_frames: u64,
        frame_step: u64,
        stride_length: u64,
        num_first_particle: u64,
        n_particles: u64,
    ) -> u64 {
        let mut length = 0;
        let size = data.data_type.get_size();

        // in C, `char` is 1 byte, so in Rust we use `u8`
        length += size_of::<u8>() * 2;
        length += size_of_val(&data.n_values_per_frame);
        length += size_of_val(&data.codec_id);

        if is_particle_data {
            length += size_of_val(&num_first_particle)
        }
        if stride_length > 1 {
            length += size_of_val(&data.first_frame_with_data) + size_of_val(&data.stride_length);
        }

        if data.codec_id != Compression::Uncompressed {
            length += size_of_val(&data.compression_multiplier);
        }

        if data.dependency & FRAME_DEPENDENT != 0 {
            length += size_of::<u8>();
        }

        if data.data_type == DataType::Char {
            let strings_3d = match &data.strings {
                Some(s) => s,
                None => unreachable!("data.strings was None"),
            };
            if is_particle_data {
                for i in 0..n_frames {
                    let first_dim_values = &strings_3d[i as usize];
                    for j in num_first_particle..n_particles {
                        let second_dim_values = &first_dim_values[j as usize];
                        for k in 0..data.n_values_per_frame {
                            length += second_dim_values[k as usize].len();
                        }
                    }
                }
            } else {
                for i in 0..n_frames {
                    let second_dim_values = &strings_3d[0][i as usize];
                    for j in 0..data.n_values_per_frame {
                        length += second_dim_values[j as usize].len() + 1;
                    }
                }
            }
        } else {
            length += size
                * usize::try_from(frame_step * n_particles).expect("usize from u64")
                * usize::try_from(data.n_values_per_frame).expect("usize from i64");
        }

        u64::try_from(length).expect("u64 from usize")
    }

    /// Write the header of a data block, regardless of its type
    fn block_header_write(&mut self, block: &mut GenBlock) {
        self.output_file_init();
        let out_file = self.output_file.as_mut().expect("init output_file");

        block.header_contents_size = block.calculate_header_len();
        utils::write_u64(
            out_file,
            block.header_contents_size,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_u64(
            out_file,
            block.block_contents_size,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_u64(
            out_file,
            block.id as u64,
            self.endianness64,
            self.output_swap64,
        );

        out_file
            .write_all(&block.md5_hash)
            .expect("able to write to output_file");
        utils::fwrite_str(out_file, block.name.as_ref().expect("block to have name"));
        utils::write_u64(
            out_file,
            block.version,
            self.endianness64,
            self.output_swap64,
        );
    }

    /// Write a general info block. This is the first block of a TNG file
    fn general_info_block_write(&mut self) {
        self.output_file_init();

        let out_file = self.output_file.as_mut().expect("init input_file");
        out_file
            .seek(SeekFrom::Start(0))
            .expect("no error handling");

        let mut block = GenBlock::new();
        block.name = Some("GENERAL INFO".to_string());
        block.id = BlockID::TrajectoryFrameSet;
        block.block_contents_size = self.general_info_block_len_calculate();
        let header_file_pos = 0;
        self.block_header_write(&mut block);

        // TODO: HASH

        let out_file = self.output_file.as_mut().expect("init input_file");
        utils::fwrite_str(out_file, &self.first_program_name);
        utils::fwrite_str(out_file, &self.last_program_name);
        utils::fwrite_str(out_file, &self.first_user_name);
        utils::fwrite_str(out_file, &self.last_user_name);
        utils::fwrite_str(out_file, &self.first_computer_name);
        utils::fwrite_str(out_file, &self.last_computer_name);
        utils::fwrite_str(out_file, &self.first_pgp_signature);
        utils::fwrite_str(out_file, &self.last_pgp_signature);
        utils::fwrite_str(out_file, &self.forcefield_name);

        utils::write_u64(out_file, self.time, self.endianness64, self.output_swap64);
        utils::write_bool(out_file, self.var_num_atoms);
        utils::write_i64(
            out_file,
            self.frame_set_n_frames,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_i64(
            out_file,
            self.first_trajectory_frame_set_output_pos,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_i64(
            out_file,
            self.last_trajectory_frame_set_output_pos,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_i64(
            out_file,
            self.medium_stride_length,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_i64(
            out_file,
            self.long_stride_length,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_i64(
            out_file,
            self.distance_unit_exponential,
            self.endianness64,
            self.output_swap64,
        );

        //TODO: HASH
    }

    /// Write a molecules block.
    fn molecules_block_write(&mut self) {
        self.output_file_init();

        let mut block = GenBlock::new();
        block.name = Some("MOLECULES".to_string());
        block.id = BlockID::Molecules;
        self.molecules_block_len_calculate();

        self.block_header_write(&mut block);

        // TODO: HASH

        let out_file = self.output_file.as_mut().expect("init output_file");
        utils::write_i64(
            out_file,
            self.n_molecules,
            self.endianness64,
            self.output_swap64,
        );

        for (molecule, mol_count) in self.molecules.iter().zip(&self.molecule_cnt_list) {
            utils::write_i64(out_file, molecule.id, self.endianness64, self.output_swap64);
            utils::fwrite_str(out_file, &molecule.name);
            utils::write_i64(
                out_file,
                molecule.quaternary_str,
                self.endianness64,
                self.output_swap64,
            );

            if !self.var_num_atoms {
                utils::write_i64(out_file, *mol_count, self.endianness64, self.output_swap64);
            }
            utils::write_i64(
                out_file,
                molecule.n_chains,
                self.endianness64,
                self.output_swap64,
            );
            utils::write_i64(
                out_file,
                molecule.n_residues,
                self.endianness64,
                self.output_swap64,
            );
            utils::write_i64(
                out_file,
                molecule.n_atoms,
                self.endianness64,
                self.output_swap64,
            );

            if molecule.n_chains > 0 {
                for chain in &molecule.chains {
                    utils::write_u64(out_file, chain.id, self.endianness64, self.output_swap64);
                    utils::fwrite_str(out_file, &chain.name);
                    utils::write_u64(
                        out_file,
                        chain.n_residues,
                        self.endianness64,
                        self.output_swap64,
                    );
                    let (start, end) = chain.residues_indices;
                    for res_index in start..end {
                        let residue = &molecule.residues[res_index];
                        utils::write_u64(
                            out_file,
                            residue.id,
                            self.endianness64,
                            self.output_swap64,
                        );
                        utils::fwrite_str(out_file, &residue.name);
                        utils::write_u64(
                            out_file,
                            residue.n_atoms,
                            self.endianness64,
                            self.output_swap64,
                        );

                        let atom_slice = &molecule.atoms[residue.n_atoms as usize
                            ..residue.n_atoms as usize + residue.atoms_offset];
                        for atom in atom_slice {
                            utils::write_i64(
                                out_file,
                                atom.id,
                                self.endianness64,
                                self.output_swap64,
                            );
                            utils::fwrite_str(out_file, &atom.name);
                            utils::fwrite_str(out_file, &atom.atom_type);
                        }
                    }
                }
            } else if molecule.n_residues > 0 {
                for residue in &molecule.residues {
                    utils::write_u64(out_file, residue.id, self.endianness64, self.output_swap64);
                    utils::fwrite_str(out_file, &residue.name);
                    utils::write_u64(
                        out_file,
                        residue.n_atoms,
                        self.endianness64,
                        self.output_swap64,
                    );
                    let atom_slice = &molecule.atoms
                        [residue.n_atoms as usize..residue.n_atoms as usize + residue.atoms_offset];
                    for atom in atom_slice {
                        utils::write_i64(out_file, atom.id, self.endianness64, self.output_swap64);
                        utils::fwrite_str(out_file, &atom.name);
                        utils::fwrite_str(out_file, &atom.atom_type);
                    }
                }
            } else {
                for atom in &molecule.atoms {
                    utils::write_i64(out_file, atom.id, self.endianness64, self.output_swap64);
                    utils::fwrite_str(out_file, &atom.name);
                    utils::fwrite_str(out_file, &atom.atom_type);
                }
            }

            utils::write_i64(
                out_file,
                molecule.n_bonds,
                self.endianness64,
                self.output_swap64,
            );

            for bond in &molecule.bonds {
                utils::write_i64(
                    out_file,
                    bond.from_atom_id,
                    self.endianness64,
                    self.output_swap64,
                );
                utils::write_i64(
                    out_file,
                    bond.to_atom_id,
                    self.endianness64,
                    self.output_swap64,
                );
            }
        }

        // TODO; HASH
    }

    fn frame_set_block_len_calculate(&self) -> u64 {
        let mut length = std::mem::size_of::<i64>() * 8;
        length += std::mem::size_of::<f64>() * 2;

        if self.var_num_atoms {
            length += std::mem::size_of::<i64>()
                * usize::try_from(self.n_molecules).expect("usize from i64");
        }

        u64::try_from(length).expect("u64 from usize")
    }

    fn frame_set_block_write(&mut self, block: &mut GenBlock) {
        self.output_file_init();

        block.name = Some("TRAJECTORY FRAME SET".to_string());
        block.id = BlockID::TrajectoryFrameSet;

        block.block_contents_size = self.frame_set_block_len_calculate();

        // TODO: hash mode - headeR_file_pos is only used for hash mode
        let header_file_pos = self
            .output_file
            .as_mut()
            .expect("init output_file")
            .stream_position()
            // TODO
            .expect("no error handling");

        self.block_header_write(block);

        // TODO: hash mode. line 3616 tng_io.c

        // let frame_set = self.current_trajectory_frame_set;
        let out_file = self.output_file.as_mut().expect("init input_file");
        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set.first_frame,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set.n_frames,
            self.endianness64,
            self.output_swap64,
        );

        if self.var_num_atoms {
            for i in 0..self.n_molecules {
                utils::write_i64(
                    out_file,
                    self.current_trajectory_frame_set.molecule_cnt_list
                        [usize::try_from(i).expect("usize from u64")],
                    self.endianness64,
                    self.output_swap64,
                );
            }
        }
        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set.next_frame_set_file_pos,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set.prev_frame_set_file_pos,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set
                .medium_stride_next_frame_set_file_pos,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set
                .medium_stride_prev_frame_set_file_pos,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set
                .long_stride_next_frame_set_file_pos,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_i64(
            out_file,
            self.current_trajectory_frame_set
                .long_stride_prev_frame_set_file_pos,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_f64(
            out_file,
            self.current_trajectory_frame_set.first_frame_time,
            self.endianness64,
            self.output_swap64,
        );

        utils::write_f64(
            out_file,
            self.time_per_frame,
            self.endianness64,
            self.output_swap64,
        );

        // TODO: hash mode tng_io.c line 3706
    }

    fn trajectory_mapping_block_len_calculate(&self, n_particles: i64) -> usize {
        std::mem::size_of::<i64>() * (2 + usize::try_from(n_particles).expect("usize from i64"))
    }

    fn tng_compress(
        &self,
        compress_algo_pos: &mut Vec<i32>,
        compress_algo_vel: &mut Vec<i32>,
        block: &GenBlock,
        n_frames: i64,
        n_particles: i64,
        data_type: &DataType,
        data: &[u8],
    ) -> Result<i64, ()> {
        let dest;

        let mut algo_find_n_frames = -1;
        if block.id != BlockID::TrajPositions || block.id != BlockID::TrajVelocities {
            eprintln!("Can only compress positions and velocities with the TNG method");
            return Err(());
        }

        if *data_type != DataType::Float || *data_type != DataType::Double {
            eprintln!("Data type not supported");
            return Err(());
        }

        if n_frames <= 0 || n_particles <= 0 {
            eprintln!("Missing frames or particles. Cannot compress data with the TNG method");
            return Err(());
        }

        let f_precision: f32 = 1.0 / (self.compression_precision as f32);
        let d_precision: f64 = 1.0 / self.compression_precision;

        if block.id == BlockID::TrajPositions {
            // If there is only one frame in this frame set and there might be more
            // do not store the algorithm as the compression algorithm, but find
            // the best one without storing it
            if n_frames == 1 && self.frame_set_n_frames > 1 {
                let nalgo = usize::try_from(tng_compress_nalgo()).expect("usize from u64");
                let mut alt_algo = vec![0; nalgo * size_of_val(&compress_algo_pos)];

                // If we have already determined the initial coding and
                // initial coding parameter do not determine them again
                if !compress_algo_pos.is_empty() {
                    alt_algo[0] = compress_algo_pos[0];
                    alt_algo[1] = compress_algo_pos[1];
                    alt_algo[2] = compress_algo_pos[2];
                    alt_algo[3] = compress_algo_pos[3];
                } else {
                    alt_algo = vec![-1; 4];
                }

                // If the initial coding and initial coding parameter are -1
                // they will be determined in tng_compress_pos/_float/
                dest = if *data_type == DataType::Float {
                    debug_assert!(
                        data.len() % 4 == 0,
                        "Float‐branch: data_bytes.len() must be exactly count * 4"
                    );

                    // TODO: maybe just re-interpret these bytes
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_le_bytes(arr));
                    }

                    tng_compress_pos_float(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        &mut alt_algo,
                    )
                } else {
                    let mut doubles = Vec::new();
                    for chunk in data.chunks_exact(8) {
                        let arr = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        doubles.push(f64::from_le_bytes(arr));
                    }
                    tng_compress_pos(
                        &doubles,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        d_precision,
                        0,
                        &mut alt_algo,
                    )
                };

                // If there had been no algorithm determined before keep the initial coding
                // and initial coding parameter so that they won't have to be determined again.
                if !compress_algo_pos.is_empty() {
                    let nalgo = usize::try_from(tng_compress_nalgo()).expect("usize from u64");
                    *compress_algo_pos = vec![0; nalgo * size_of_val(compress_algo_pos)];
                    compress_algo_pos[0] = alt_algo[0];
                    compress_algo_pos[1] = alt_algo[1];
                    compress_algo_pos[2] = -1;
                    compress_algo_pos[3] = -1;
                }
            // TODO: is it a bug in the original code that it checks twice for the compress_algo_pos?
            } else if !compress_algo_pos.is_empty()
                || compress_algo_pos[2] == -1
                || compress_algo_pos[2] == -1
            {
                algo_find_n_frames = if n_frames > 6 { 5 } else { n_frames };

                // If the algorithm parameters are -1 they will be determined during the compression.
                if !compress_algo_pos.is_empty() {
                    let nalgo = usize::try_from(tng_compress_nalgo()).expect("usize from u64");
                    *compress_algo_pos = vec![0; nalgo * size_of_val(compress_algo_pos)];
                    compress_algo_pos[0] = -1;
                    compress_algo_pos[1] = -1;
                    compress_algo_pos[2] = -1;
                    compress_algo_pos[3] = -1;
                }

                dest = if *data_type == DataType::Float {
                    debug_assert!(
                        data.len().is_multiple_of(4),
                        "Float‐branch: data_bytes.len() must be exactly count * 4"
                    );

                    // TODO: maybe just re-interpret these bytes
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_le_bytes(arr));
                    }

                    let mut return_dest = tng_compress_pos_float(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(algo_find_n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        compress_algo_pos,
                    );
                    if algo_find_n_frames < n_frames {
                        return_dest = tng_compress_pos_float(
                            &floats,
                            usize::try_from(n_particles).expect("usize from i64"),
                            usize::try_from(n_frames).expect("usize from i64"),
                            f_precision,
                            0,
                            compress_algo_pos,
                        );
                    }
                    return_dest
                } else {
                    let mut doubles = Vec::new();
                    for chunk in data.chunks_exact(8) {
                        let arr = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        doubles.push(f64::from_le_bytes(arr));
                    }
                    let mut return_dest = tng_compress_pos(
                        &doubles,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(algo_find_n_frames).expect("usize from i64"),
                        d_precision,
                        0,
                        compress_algo_pos,
                    );

                    if algo_find_n_frames < n_frames {
                        return_dest = tng_compress_pos(
                            &doubles,
                            usize::try_from(n_particles).expect("usize from i64"),
                            usize::try_from(n_frames).expect("usize from i64"),
                            d_precision,
                            0,
                            compress_algo_pos,
                        );
                    }
                    return_dest
                };
            } else {
                dest = if *data_type == DataType::Float {
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_le_bytes(arr));
                    }
                    tng_compress_vel_float(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        compress_algo_pos,
                    )
                } else {
                    let mut doubles = Vec::new();
                    for chunk in data.chunks_exact(8) {
                        let arr = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        doubles.push(f64::from_le_bytes(arr));
                    }
                    tng_compress_vel(
                        &doubles,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        d_precision,
                        0,
                        compress_algo_pos,
                    )
                }
            }
        } else if block.id == BlockID::TrajVelocities {
            // If there is only one frame in this frame set and there might be more
            // do not store the algorithm as the compression algorithm, but find
            // the best one without storing it
            if n_frames == 1 && self.frame_set_n_frames > 1 {
                let nalgo = usize::try_from(tng_compress_nalgo()).expect("usize from u64");
                let mut alt_algo = vec![0; nalgo * size_of_val(compress_algo_vel)];

                // If we have already determined the initial coding and
                // initial coding parameter do not determine them again
                if !compress_algo_vel.is_empty() {
                    alt_algo[0] = compress_algo_vel[0];
                    alt_algo[1] = compress_algo_vel[1];
                    alt_algo[2] = compress_algo_vel[2];
                    alt_algo[3] = compress_algo_vel[3];
                } else {
                    alt_algo = vec![-1; 4];
                }

                // If the initial coding and initial coding parameter are -1
                // they will be determined in tng_compress_pos/_float/.
                dest = if *data_type == DataType::Float {
                    // TODO: maybe just re-interpret these bytes
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_le_bytes(arr));
                    }
                    tng_compress_vel_float(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        &mut alt_algo,
                    )
                } else {
                    let mut doubles = Vec::new();
                    for chunk in data.chunks_exact(8) {
                        let arr = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        doubles.push(f64::from_le_bytes(arr));
                    }
                    tng_compress_vel(
                        &doubles,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(algo_find_n_frames).expect("usize from i64"),
                        d_precision,
                        0,
                        &mut alt_algo,
                    )
                };
                // If there had been no algorithm determined before keep the initial coding
                // and initial coding parameter so that they won't have to be determined again
                if compress_algo_vel.is_empty() {
                    let nalgo = usize::try_from(tng_compress_nalgo()).expect("usize from u64");
                    *compress_algo_vel = vec![0; nalgo * size_of_val(compress_algo_vel)];
                    compress_algo_vel[0] = alt_algo[0];
                    compress_algo_vel[1] = alt_algo[1];
                    compress_algo_vel[2] = -1;
                    compress_algo_vel[3] = -1;
                }
            // TODO: is it a bug in the original code that it checks twice for the compress_algo_vel?
            } else if compress_algo_vel.is_empty()
                || compress_algo_vel[2] == -1
                || compress_algo_vel[2] == -1
            {
                algo_find_n_frames = if n_frames > 6 { 5 } else { n_frames };

                // If the algorithm parameters are -1 they will be determined during the compression
                if compress_algo_vel.is_empty() {
                    let nalgo = usize::try_from(tng_compress_nalgo()).expect("usize from u64");
                    *compress_algo_vel = vec![-1; nalgo * size_of_val(compress_algo_vel)];
                }

                dest = if *data_type == DataType::Float {
                    // TODO: maybe just re-interpret these bytes
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_le_bytes(arr));
                    }
                    let mut return_dest = tng_compress_vel_float(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(algo_find_n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        compress_algo_vel,
                    );
                    if algo_find_n_frames < n_frames {
                        return_dest = tng_compress_vel_float(
                            &floats,
                            usize::try_from(n_particles).expect("usize from i64"),
                            usize::try_from(n_frames).expect("usize from i64"),
                            f_precision,
                            0,
                            compress_algo_vel,
                        );
                    }
                    return_dest
                } else {
                    let mut doubles = Vec::new();
                    for chunk in data.chunks_exact(8) {
                        let arr = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        doubles.push(f64::from_le_bytes(arr));
                    }
                    let mut return_dest = tng_compress_vel(
                        &doubles,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(algo_find_n_frames).expect("usize from i64"),
                        d_precision,
                        0,
                        compress_algo_vel,
                    );
                    if algo_find_n_frames < n_frames {
                        return_dest = tng_compress_vel(
                            &doubles,
                            usize::try_from(n_particles).expect("usize from i64"),
                            usize::try_from(n_frames).expect("usize from i64"),
                            d_precision,
                            0,
                            compress_algo_vel,
                        );
                    }
                    return_dest
                };
            } else {
                dest = if *data_type == DataType::Float {
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_le_bytes(arr));
                    }
                    tng_compress_vel_float(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        compress_algo_vel,
                    )
                } else {
                    let mut doubles = Vec::new();
                    for chunk in data.chunks_exact(8) {
                        let arr = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        doubles.push(f64::from_le_bytes(arr));
                    }
                    tng_compress_vel(
                        &doubles,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        d_precision,
                        0,
                        compress_algo_vel,
                    )
                }
            }
        } else {
            error!("Can only compress positions and velocities using TNG-MF1 algorithms");
            return Err(());
        }

        Ok(i64::try_from(dest.unwrap().len()).expect("i64"))
    }

    fn tng_uncompress() {
        unimplemented!("tng/lib_io.c 4326")
    }

    /// Write a data block (particle or non-particle data)
    fn data_block_write(
        &mut self,
        // output_file: &mut Option<File>,
        block: &mut GenBlock,
        block_index: usize,
        is_particle_data: bool,
        mapping: &Option<ParticleMapping>,
        hash_mode: bool,
    ) {
        // If we have already started writing frame sets it is too late to write
        // non-trajectory data blocks
        let is_trajectory_block = self.current_trajectory_frame_set_output_file_pos > 0;

        self.output_file_init();

        let slot = match (is_particle_data, is_trajectory_block) {
            (true, true) => Slot::TrParticle,
            (true, false) => Slot::NonTrParticle,
            (false, true) => Slot::Tr,
            (false, false) => Slot::NonTr,
        };

        // helper: one-shot mutable access
        let data_mut = match slot {
            Slot::TrParticle => {
                &mut self.current_trajectory_frame_set.tr_particle_data[block_index]
            }
            Slot::NonTrParticle => &mut self.non_tr_particle_data[block_index],
            Slot::Tr => &mut self.current_trajectory_frame_set.tr_data[block_index],
            Slot::NonTr => &mut self.non_tr_data[block_index],
        };

        let stride_length = {
            if is_trajectory_block
                && data_mut.first_frame_with_data < self.current_trajectory_frame_set.first_frame
            {
                return;
            }
            data_mut.stride_length.max(1)
        };

        // if is_particle_data {
        //     if is_trajectory_block {
        //         data = &mut self.current_trajectory_frame_set.tr_particle_data[block_index];

        //         // If this data block has not had any data added in this frame set
        //         // do not write it
        //         if data.first_frame_with_data < self.current_trajectory_frame_set.first_frame {
        //             return;
        //         }

        //         stride_length = max(1, data.stride_length);
        //     } else {
        //         data = &mut self.non_tr_particle_data[block_index];
        //         stride_length = 1;
        //     }
        // } else if is_trajectory_block {
        //     data = &mut self.current_trajectory_frame_set.tr_data[block_index];

        //     // If this data block has not had any data added in this frame set
        //     // do not write it
        //     if data.first_frame_with_data < self.current_trajectory_frame_set.first_frame {
        //         return;
        //     }

        //     stride_length = max(1, data.stride_length);
        // } else {
        //     data = &mut self.non_tr_data[block_index];
        //     stride_length = 1;
        // }

        let size = data_mut.data_type.get_size();
        block.name = Some(data_mut.block_name.clone());
        block.id = data_mut.block_id;

        // If writing frame independent data data->n_frames is 0, but n_frames
        // is used for the loop writing the data (and reserving memory) and needs
        // to be at least 1
        let mut n_frames = max(1, data_mut.n_frames);

        if is_trajectory_block {
            // If the frame is finished before writing the full number of frames
            // make sure the data block is not longer than the frame set
            n_frames = min(n_frames, self.current_trajectory_frame_set.n_frames);
            n_frames -=
                data_mut.first_frame_with_data - self.current_trajectory_frame_set.first_frame;
        }

        let mut frame_step = (n_frames - 1) / stride_length + 1;

        let compression_precision = self.compression_precision;
        match data_mut.codec_id {
            Compression::XTC => unimplemented!("XTC compression is not yet implemented"),
            // TNG compression will use compression precision to get integers from
            // floating point data. The compression multiplier stores that information
            // to be able to return the precision of the compressed data
            Compression::TNG => {
                data_mut.compression_multiplier = compression_precision;
            }
            // Uncompressed data blocks do not use compression multipliers at all.
            // GZip compression does not need it either
            Compression::Uncompressed | Compression::GZip => {
                data_mut.compression_multiplier = 1.0;
            }
        }

        let mut n_particles = -1;
        let mut num_first_particle = -1;
        if data_mut.dependency & PARTICLE_DEPENDENT != 0 {
            if let Some(mapping) = mapping
                && mapping.n_particles != 0
            {
                n_particles = mapping.n_particles;
                num_first_particle = mapping.num_first_particle;
            } else {
                num_first_particle = 0;
                if self.var_num_atoms {
                    n_particles = self.current_trajectory_frame_set.n_particles;
                } else {
                    n_particles = self.n_particles;
                }
            }
        }

        // TODO: we probably want to avoid cloning the data blocks
        let mut cloned_data = data_mut.clone();
        if data_mut.dependency & PARTICLE_DEPENDENT != 0 {
            block.block_contents_size = Self::data_block_len_calculate(
                &cloned_data,
                true,
                u64::try_from(n_frames).expect("(u64 from i64)"),
                u64::try_from(frame_step).expect("(u64 from i64)"),
                u64::try_from(stride_length).expect("(u64 from i64)"),
                u64::try_from(num_first_particle).expect("u64 from i64"),
                u64::try_from(n_particles).expect("u64 to i64"),
            );
        } else {
            block.block_contents_size = Self::data_block_len_calculate(
                &cloned_data,
                false,
                u64::try_from(n_frames).expect("(u64 from i64)"),
                u64::try_from(frame_step).expect("(u64 from i64)"),
                u64::try_from(stride_length).expect("(u64 from i64)"),
                0,
                1,
            );
        }

        let header_file_pos = self.get_output_file_position();

        self.block_header_write(block);

        // TODO: hash mode

        let out_file = self.output_file.as_mut().expect("init output_file");
        utils::write_u8(out_file, cloned_data.data_type as u8);
        utils::write_u8(out_file, cloned_data.dependency);

        if cloned_data.dependency & FRAME_DEPENDENT != 0 {
            let temp = if stride_length > 1 { 1 } else { 0 };
            utils::write_u8(out_file, temp);
        }

        utils::write_i64(
            out_file,
            cloned_data.n_values_per_frame,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_u64(
            out_file,
            cloned_data.codec_id as u64,
            self.endianness64,
            self.output_swap64,
        );

        if cloned_data.codec_id != Compression::Uncompressed {
            utils::write_f64(
                out_file,
                cloned_data.compression_multiplier,
                self.endianness64,
                self.output_swap64,
            );
        }

        if cloned_data.n_frames > 0 && stride_length > 1 {
            // FIXME(from c): first_frame_with_data is not reliably set
            if cloned_data.first_frame_with_data == 0 {
                cloned_data.first_frame_with_data = self.current_trajectory_frame_set.first_frame;
            }
            utils::write_i64(
                out_file,
                cloned_data.first_frame_with_data,
                self.endianness64,
                self.output_swap64,
            );
            utils::write_i64(
                out_file,
                stride_length,
                self.endianness64,
                self.output_swap64,
            );
        }

        if cloned_data.dependency & PARTICLE_DEPENDENT != 0 {
            utils::write_i64(
                out_file,
                num_first_particle,
                self.endianness64,
                self.output_swap64,
            );
            utils::write_i64(out_file, n_particles, self.endianness64, self.output_swap64);
        }

        if cloned_data.data_type == DataType::Char {
            if let Some(strings_3d) = cloned_data.strings {
                if cloned_data.dependency & PARTICLE_DEPENDENT != 0 {
                    for i in 0..frame_step {
                        let first_dim_values = &strings_3d[i as usize];
                        for j in num_first_particle..num_first_particle + n_particles {
                            let second_dim_values = &first_dim_values[j as usize];
                            for k in 0..cloned_data.n_values_per_frame {
                                utils::fwrite_str(out_file, &second_dim_values[k as usize]);
                            }
                        }
                    }
                } else {
                    for i in 0..frame_step {
                        for j in 0..cloned_data.n_values_per_frame {
                            utils::fwrite_str(out_file, &strings_3d[0][i as usize][j as usize]);
                        }
                    }
                }
            }
        } else {
            let mut full_data_len = size
                * usize::try_from(frame_step * cloned_data.n_values_per_frame)
                    .expect("usize from i64");
            if cloned_data.dependency & PARTICLE_DEPENDENT != 0 {
                full_data_len *= usize::try_from(n_particles).expect("usize from i64");
            }

            let mut contents = vec![0; full_data_len];
            if let Some(values) = cloned_data.values {
                contents = values.clone();

                // If writing TNG compressed data the endianness is taken into account by
                // the compression routines. TNG compressed data is always written as little endian
                if cloned_data.codec_id != Compression::Uncompressed {
                    match cloned_data.data_type {
                        DataType::Float => match cloned_data.codec_id {
                            Compression::Uncompressed | Compression::GZip => {
                                if let Some(output_swap32) = self.output_swap32 {
                                    debug_assert!(full_data_len % 4 == 0);

                                    for chunk in contents.chunks_exact_mut(size) {
                                        let mut val = u32::from_ne_bytes(chunk.try_into().unwrap());
                                        output_swap32(self.endianness32, &mut val);

                                        chunk.copy_from_slice(&val.to_ne_bytes());
                                    }
                                }
                            }
                            Compression::XTC | Compression::TNG => {
                                let multiplier = cloned_data.compression_multiplier;
                                if (multiplier - 1.0).abs() > 0.00001
                                    || self.output_swap32.is_some()
                                {
                                    for chunk in contents.chunks_exact_mut(size) {
                                        let orig_bits =
                                            u32::from_ne_bytes(chunk.try_into().unwrap());
                                        let mut val = f32::from_bits(orig_bits);
                                        val *= multiplier as f32;

                                        let mut new_bits = val.to_bits();
                                        if let Some(output_swap32) = self.output_swap32 {
                                            output_swap32(self.endianness32, &mut new_bits)
                                        }
                                        chunk.copy_from_slice(&new_bits.to_ne_bytes());
                                    }
                                }
                            }
                        },
                        DataType::Int => {
                            if let Some(output_swap64) = self.output_swap64 {
                                debug_assert!(full_data_len % 8 == 0);

                                for chunk in contents.chunks_exact_mut(size) {
                                    let mut val = u64::from_ne_bytes(chunk.try_into().unwrap());
                                    output_swap64(self.endianness64, &mut val);

                                    chunk.copy_from_slice(&val.to_ne_bytes());
                                }
                            }
                        }
                        DataType::Double => match cloned_data.codec_id {
                            Compression::Uncompressed | Compression::GZip => {
                                if let Some(output_swap64) = self.output_swap64 {
                                    debug_assert!(full_data_len % 8 == 0);

                                    for chunk in contents.chunks_exact_mut(size) {
                                        let mut val = u64::from_ne_bytes(chunk.try_into().unwrap());
                                        output_swap64(self.endianness64, &mut val);

                                        chunk.copy_from_slice(&val.to_ne_bytes());
                                    }
                                }
                            }
                            Compression::XTC | Compression::TNG => {
                                let multiplier = cloned_data.compression_multiplier;
                                if (multiplier - 1.0).abs() > 0.00001
                                    || self.output_swap64.is_some()
                                {
                                    for chunk in contents.chunks_exact_mut(size) {
                                        let orig_bits =
                                            u64::from_ne_bytes(chunk.try_into().unwrap());
                                        let mut val = f64::from_bits(orig_bits);
                                        val *= multiplier;

                                        let mut new_bits = val.to_bits();
                                        if let Some(output_swap64) = self.output_swap64 {
                                            output_swap64(self.endianness64, &mut new_bits)
                                        }
                                        chunk.copy_from_slice(&new_bits.to_ne_bytes());
                                    }
                                }
                            }
                        },
                        DataType::Char => {}
                    }
                }
            } else {
                // the c code fills `contents` with 0, but we've already done that
            }

            let mut block_data_len = full_data_len;

            match cloned_data.codec_id {
                Compression::XTC => {
                    warn!("XTC Compression not implemented yet");
                    cloned_data.codec_id = Compression::Uncompressed;
                }
                Compression::TNG => {
                    // to avoid overlapping borrows, we mem::take and put them back afterwards
                    let mut compress_algo_pos = std::mem::take(&mut self.compress_algo_pos);
                    let mut compress_algo_vel = std::mem::take(&mut self.compress_algo_vel);
                    match self.tng_compress(
                        &mut compress_algo_pos,
                        &mut compress_algo_vel,
                        block,
                        frame_step,
                        n_particles,
                        &cloned_data.data_type,
                        &contents,
                    ) {
                        Ok(compressed_len) => {
                            block_data_len = usize::try_from(compressed_len).expect("usize")
                        }
                        Err(_) => {
                            error!("Could not write TNG compressed block data.");
                            // TODO: If critical (when?), we should panic as c does

                            // Set the data again, but with no compression (to write only the relevant data)
                            // Reborrow `data_mut`
                            // TODO: is there a way to get rid of this reborrow?
                            let data_mut = match slot {
                                Slot::TrParticle => {
                                    &mut self.current_trajectory_frame_set.tr_particle_data
                                        [block_index]
                                }
                                Slot::NonTrParticle => &mut self.non_tr_particle_data[block_index],
                                Slot::Tr => {
                                    &mut self.current_trajectory_frame_set.tr_data[block_index]
                                }
                                Slot::NonTr => &mut self.non_tr_data[block_index],
                            };
                            data_mut.codec_id = Compression::Uncompressed;
                            self.data_block_write(
                                // output_file,
                                block,
                                block_index,
                                is_particle_data,
                                mapping,
                                hash_mode,
                            );
                        }
                    }
                    self.compress_algo_pos = compress_algo_pos;
                    self.compress_algo_vel = compress_algo_vel;
                }
                Compression::GZip => match Self::gzip_compress(&contents, contents.len()) {
                    Ok(compressed_len) => block_data_len = compressed_len,
                    Err(_) => {
                        error!("Could not write gzipped block data.");
                        // TODO: is there a way to get rid of this reborrow?
                        let data_mut = match slot {
                            Slot::TrParticle => {
                                &mut self.current_trajectory_frame_set.tr_particle_data[block_index]
                            }
                            Slot::NonTrParticle => &mut self.non_tr_particle_data[block_index],
                            Slot::Tr => &mut self.current_trajectory_frame_set.tr_data[block_index],
                            Slot::NonTr => &mut self.non_tr_data[block_index],
                        };
                        data_mut.codec_id = Compression::Uncompressed;
                    }
                },
                Compression::Uncompressed => {
                    // this is silently skipped in the C code
                }
            }

            if block_data_len != full_data_len {
                block.block_contents_size -=
                    u64::try_from(full_data_len - block_data_len).expect("u64 to usize");

                let file = self.output_file.as_mut().expect("init output_file");

                // c version is ftello
                let curr_file_pos = file.stream_position().expect("no error handling");

                // c version is fseeko
                let offset =
                    header_file_pos + std::mem::size_of_val(&block.header_contents_size) as u64;
                file.seek(SeekFrom::Start(offset))
                    .expect("no error handling");

                utils::write_u64(
                    file,
                    block.block_contents_size,
                    self.endianness64,
                    self.output_swap64,
                );
                file.seek(SeekFrom::Start(curr_file_pos))
                    .expect("no error handling");
            }
            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .write_all(&contents[..block_data_len])
                .expect("Could not write all block data.");
            // TODO: hash mode lib/tng_io.c 5851
        }

        // if hash_mode == TNG_USE_HASH {
        //     unimplemented!("tng/lib_io.c 5859")
        // }
        // frame_set
        self.current_trajectory_frame_set.n_written_frames *=
            self.current_trajectory_frame_set.n_unwritten_frames;
        self.current_trajectory_frame_set.n_unwritten_frames = 0;
    }

    pub fn file_headers_write(&mut self, hash_mode: bool) -> Result<(), std::io::Error> {
        let mut temp_pos = None;
        let mut total_len = 0;
        self.output_file_init();

        if self.n_trajectory_frame_sets > 0 {
            let orig_len = self.file_headers_len_get();

            let mut block = GenBlock::new();
            block.name = Some("GENERAL INFO".to_string());
            total_len += block.calculate_header_len();
            total_len += self.general_info_block_len_calculate();

            block.name = Some("MOLECULES".to_string());
            total_len += block.calculate_header_len();
            total_len += self.molecules_block_len_calculate();

            for i in 0..self.n_data_blocks {
                let data = &self.non_tr_data[i];
                block.name = Some(data.block_name.clone());
                total_len += block.calculate_header_len();
                total_len += Self::data_block_len_calculate(data, false, 1, 1, 1, 0, 1);
            }

            for i in 0..self.n_particle_data_blocks {
                let data = &self.non_tr_particle_data[i];
                block.name = Some(data.block_name.clone());
                total_len += block.calculate_header_len();
                total_len += Self::data_block_len_calculate(data, true, 1, 1, 1, 0, 1);
            }

            let orig_len = u64::try_from(orig_len).expect("u64 from usize");
            if total_len > orig_len {
                self.migrate_data_in_file(
                    i64::try_from(orig_len + 1).expect("i64 from usize"),
                    i64::try_from(total_len - orig_len).expect("i64 from usize"),
                    hash_mode,
                );
                self.last_trajectory_frame_set_input_pos =
                    self.last_trajectory_frame_set_output_pos;
            }

            self.reread_frame_set_at_file_pos(
                u64::try_from(self.last_trajectory_frame_set_input_pos).expect("u64 from i64"),
            );

            // In order to write non-trajectory data the current_trajectory_frame_set_output_file_pos
            // must temporarily be reset
            temp_pos = Some(self.current_trajectory_frame_set_output_file_pos);
            self.current_trajectory_frame_set_output_file_pos = -1;
        }

        self.general_info_block_write();

        self.molecules_block_write();

        // FIXME(from c): Currently writing non-trajectory data blocks here.
        // Should perhaps be moved
        let mut block = GenBlock::new();

        for i in 0..self.n_data_blocks {
            block.id = self.non_tr_data[i].block_id;
            self.data_block_write(&mut block, i, false, &None, hash_mode)
        }

        for i in 0..self.n_particle_data_blocks {
            block.id = self.non_tr_particle_data[i].block_id;
            self.data_block_write(&mut block, i, true, &None, hash_mode);
        }

        // Continue writing at the end of the file
        self.output_file
            .as_mut()
            .expect("init output_file")
            .seek(SeekFrom::End(0))?;
        if let Some(temp_pos) = temp_pos {
            self.current_trajectory_frame_set_output_file_pos = temp_pos;
        }

        Ok(())
    }

    fn file_pos_of_subsequent_trajectory_block_get(&mut self) -> i64 {
        let orig_pos = self.get_input_file_position();
        let curr_frame_set_pos =
            u64::try_from(self.current_trajectory_frame_set_input_file_pos).expect("u64 from i64");

        let mut pos = self.first_trajectory_frame_set_input_pos;

        if pos <= 0 {
            return pos;
        }

        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(u64::try_from(pos).expect("u64 from i64")))
            .expect("no error handling");
        let mut block = GenBlock::new();

        // Read block headers first to see that a frame set block is found
        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("Cannot read block header at pos {pos}");
        }

        self.block_read_next(&mut block);

        // Update `pos` if this is the earliest frame set so far (after `orig_pos`)
        if self.current_trajectory_frame_set_input_file_pos < pos
            && self.current_trajectory_frame_set_input_file_pos
                > i64::try_from(orig_pos).expect("i64 from u64")
        {
            pos = self.current_trajectory_frame_set_input_file_pos;
        }

        // Re-read the frame set that used to be the current one
        self.reread_frame_set_at_file_pos(curr_frame_set_pos);
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(u64::try_from(pos).expect("u64 from i64")))
            .expect("no error handling");

        pos
    }

    fn length_of_current_frame_set_contents_get(&mut self) -> i64 {
        let orig_pos = self.get_input_file_position();
        let curr_frame_set_pos =
            u64::try_from(self.current_trajectory_frame_set_input_file_pos).expect("u64 from i64");
        let mut len = 0;

        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(curr_frame_set_pos))
            .expect("no error handling");
        let mut block = GenBlock::new();

        // Read block headers first to see that a frame set block is found
        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("Cannot read block header at pos {curr_frame_set_pos}");
        }

        // Read the headers of all blocks in the frame set (not the actual contents of them)
        loop {
            self.input_file
                .as_mut()
                .expect("init input_file")
                .seek(SeekFrom::Start(curr_frame_set_pos))
                .expect("no error handling");

            len += block.header_contents_size + block.block_contents_size;
            if len >= self.input_file_len {
                break;
            }
            self.block_header_read(&mut block);
            if block.id == BlockID::TrajectoryFrameSet {
                break;
            }
        }

        self.reread_frame_set_at_file_pos(curr_frame_set_pos);

        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(orig_pos))
            .expect("no error handling");
        i64::try_from(len).expect("i64 from u64")
    }

    /// Update the frame set pointers in the current frame set block, already
    /// written to disk. It also updates the pointers of the blocks pointing to
    /// the current frame set block
    fn frame_set_pointers_update(&mut self, _hash_mode: bool) {
        self.output_file_init();
        let mut block = GenBlock::new();
        let temp_input_file = self
            .input_file
            .as_mut()
            .expect("init input_file")
            .try_clone()
            .expect("able to clone file");

        let output_file_pos = self.get_output_file_position();
        let out_file = self.output_file.as_mut().expect("init output_file");
        self.input_file = Some(out_file.try_clone().expect("able to clone output file"));
        let pos = self.current_trajectory_frame_set_output_file_pos;

        // Update next frame set
        if self.current_trajectory_frame_set.next_frame_set_file_pos > 0 {
            out_file
                .seek(SeekFrom::Start(
                    u64::try_from(self.current_trajectory_frame_set.next_frame_set_file_pos)
                        .expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);
            let contents_start_pos = self.get_output_file_position();

            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .seek(SeekFrom::Current(
                    i64::try_from(
                        block.block_contents_size
                            - u64::try_from(5 * size_of::<i64>() + 2 * size_of::<f64>())
                                .expect("u64 from usize"),
                    )
                    .expect("i64 from u64"),
                ))
                .expect("no error handling");

            if let Some(swap_fn) = self.input_swap64 {
                swap_fn(
                    self.endianness64,
                    &mut u64::try_from(pos).expect("u64 from i64"),
                );
            }
            out_file
                .write_all(&pos.to_ne_bytes())
                .expect("write to out");
        }

        // Update previous frame set
        let out_file = self.output_file.as_mut().expect("init output_file");
        if self.current_trajectory_frame_set.prev_frame_set_file_pos > 0 {
            out_file
                .seek(SeekFrom::Start(
                    u64::try_from(self.current_trajectory_frame_set.prev_frame_set_file_pos)
                        .expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);

            let contents_start_pos = self.get_output_file_position();

            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .seek(SeekFrom::Current(
                    i64::try_from(
                        block.block_contents_size
                            - u64::try_from(6 * size_of::<i64>() + 2 * size_of::<f64>())
                                .expect("u64 from usize"),
                    )
                    .expect("i64 from u64"),
                ))
                .expect("no error handling");
            if let Some(swap_fn) = self.input_swap64 {
                swap_fn(
                    self.endianness64,
                    &mut u64::try_from(pos).expect("u64 from i64"),
                )
            }
            out_file
                .write_all(&pos.to_ne_bytes())
                .expect("write to out");
        }

        // Update the frame set one medium stride step after
        let out_file = self.output_file.as_mut().expect("init output_file");
        if self
            .current_trajectory_frame_set
            .medium_stride_next_frame_set_file_pos
            > 0
        {
            out_file
                .seek(SeekFrom::Start(
                    u64::try_from(
                        self.current_trajectory_frame_set
                            .medium_stride_next_frame_set_file_pos,
                    )
                    .expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);
            let contents_start_pos = self.get_output_file_position();

            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .seek(SeekFrom::Current(
                    i64::try_from(
                        block.block_contents_size
                            - u64::try_from(3 * size_of::<i64>() + 2 * size_of::<f64>())
                                .expect("u64 from usize"),
                    )
                    .expect("i64 from u64"),
                ))
                .expect("no error handling");
            if let Some(swap_fn) = self.input_swap64 {
                swap_fn(
                    self.endianness64,
                    &mut u64::try_from(pos).expect("u64 from i64"),
                )
            }
            out_file
                .write_all(&pos.to_ne_bytes())
                .expect("write to out");

            // TODO: hash_mode
        }

        // Update the frame set one medium stride before
        let out_file = self.output_file.as_mut().expect("init output_file");
        if self
            .current_trajectory_frame_set
            .medium_stride_prev_frame_set_file_pos
            > 0
        {
            out_file
                .seek(SeekFrom::Start(
                    u64::try_from(
                        self.current_trajectory_frame_set
                            .medium_stride_prev_frame_set_file_pos,
                    )
                    .expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);

            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .seek(SeekFrom::Current(
                    i64::try_from(
                        block.block_contents_size
                            - u64::try_from(4 * size_of::<i64>() + 2 * size_of::<f64>())
                                .expect("u64 from usize"),
                    )
                    .expect("i64 from u64"),
                ))
                .expect("no error handling");
            if let Some(swap_fn) = self.input_swap64 {
                swap_fn(
                    self.endianness64,
                    &mut u64::try_from(pos).expect("u64 from i64"),
                )
            }
            out_file
                .write_all(&pos.to_ne_bytes())
                .expect("write to out");

            // TODO: hash_mode
        }

        // Update the frame set one long stride after
        let out_file = self.output_file.as_mut().expect("init output_file");
        if self
            .current_trajectory_frame_set
            .long_stride_next_frame_set_file_pos
            > 0
        {
            out_file
                .seek(SeekFrom::Start(
                    u64::try_from(
                        self.current_trajectory_frame_set
                            .long_stride_next_frame_set_file_pos,
                    )
                    .expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);

            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .seek(SeekFrom::Current(
                    i64::try_from(
                        block.block_contents_size
                            - u64::try_from(1 * size_of::<i64>() + 2 * size_of::<f64>())
                                .expect("u64 from usize"),
                    )
                    .expect("i64 from u64"),
                ))
                .expect("no error handling");
            if let Some(swap_fn) = self.input_swap64 {
                swap_fn(
                    self.endianness64,
                    &mut u64::try_from(pos).expect("u64 from i64"),
                )
            }
            out_file
                .write_all(&pos.to_ne_bytes())
                .expect("write to out");

            // TODO: hash_mode
        }

        // Update the frame set one long stride before
        let out_file = self.output_file.as_mut().expect("init output_file");
        if self
            .current_trajectory_frame_set
            .long_stride_prev_frame_set_file_pos
            > 0
        {
            out_file
                .seek(SeekFrom::Start(
                    u64::try_from(
                        self.current_trajectory_frame_set
                            .long_stride_prev_frame_set_file_pos,
                    )
                    .expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);

            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .seek(SeekFrom::Current(
                    i64::try_from(
                        block.block_contents_size
                            - u64::try_from(2 * size_of::<i64>() + 2 * size_of::<f64>())
                                .expect("u64 from usize"),
                    )
                    .expect("i64 from u64"),
                ))
                .expect("no error handling");
            if let Some(swap_fn) = self.input_swap64 {
                swap_fn(
                    self.endianness64,
                    &mut u64::try_from(pos).expect("u64 from i64"),
                )
            }
            out_file
                .write_all(&pos.to_ne_bytes())
                .expect("write to out");

            // TODO: hash_mode
        }

        let out_file = self.output_file.as_mut().expect("init output_file");
        out_file
            .seek(SeekFrom::Start(output_file_pos))
            .expect("no error handling");
        self.input_file = Some(temp_input_file.try_clone().expect("able to clone file"));
    }

    /// Migrate a whole frame set from one position in the file to another.
    fn frame_set_complete_migrate(
        &mut self,
        block_start_pos: i64,
        block_len: usize,
        new_pos: u64,
        hash_mode: bool,
    ) {
        self.input_file_init();
        let inp_file = self.input_file.as_mut().expect("init input_file");
        let out_file = self.output_file.as_mut().expect("init input_file");
        inp_file
            .seek(SeekFrom::Start(
                u64::try_from(block_start_pos).expect("u64 from i64"),
            ))
            .expect("no error handling");
        let mut contents: Vec<u8> = vec![0u8; block_len];
        inp_file.read_exact(&mut contents);
        out_file
            .seek(SeekFrom::Start(new_pos))
            .expect("no error handling");
        out_file.write_all(&contents);

        self.current_trajectory_frame_set_output_file_pos =
            i64::try_from(new_pos).expect("i64 from u64");
        if is_same_file(
            self.output_file.as_ref().expect("output file set"),
            self.input_file.as_ref().expect("input file set"),
        )
        .is_ok_and(|x| x)
        {
            self.current_trajectory_frame_set_input_file_pos =
                i64::try_from(new_pos).expect("i64 from u64");
        }

        self.frame_set_pointers_update(hash_mode);
    }

    /// Migrate data blocks in the file to make room for new data in a block. This
    /// is required e.g. when adding data to a block or extending strings in a
    /// block.
    fn migrate_data_in_file(&mut self, start_pos: i64, offset: i64, hash_mode: bool) {
        if offset <= 0 {
            return;
        }

        let traj_start_pos = self.file_pos_of_subsequent_trajectory_block_get();

        self.current_trajectory_frame_set_input_file_pos = traj_start_pos;

        let mut empty_space = traj_start_pos - (start_pos - 1);

        if empty_space >= offset {
            return;
        }

        let orig_file_pos = self.get_input_file_position();
        let mut block = GenBlock::new();

        while empty_space < offset {
            self.input_file
                .as_mut()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(traj_start_pos).expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);

            let frame_set_length = self.length_of_current_frame_set_contents_get();
            self.frame_set_complete_migrate(
                traj_start_pos,
                usize::try_from(frame_set_length).expect("usize from u64"),
                self.input_file_len,
                hash_mode,
            );

            empty_space += frame_set_length
        }
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(orig_file_pos))
            .expect("no error handling");
    }

    fn file_headers_len_get(&mut self) -> usize {
        self.input_file_init();

        let mut len = 0;
        let orig_pos = self.get_input_file_position();
        let mut block = GenBlock::new();

        // Read through the headers of non-trajectory blocks (they come before the
        // trajectory blocks in the file)
        self.block_header_read(&mut block);
        while len < self.input_file_len
            && block.id != BlockID::Unknown
            && block.id != BlockID::TrajectoryFrameSet
        {
            len += block.header_contents_size + block.block_contents_size;
            self.input_file
                .as_mut()
                .expect("init input_file")
                .seek(SeekFrom::Current(
                    i64::try_from(block.block_contents_size).expect("i64 from u64"),
                ))
                .expect("no error handling");
        }
        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(orig_pos))
            .expect("no error handling");

        usize::try_from(len).expect("usize from u64")
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
            atom_residue_index = Some(mol.residues[residue_idx].id + offset as u64);
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

    /// Get the atom type of real particle number (number in mol system).
    pub fn atom_type_of_particle_nr_get(&self, nr: i64) -> String {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut atom_type = String::new();
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            let atom = &mol.atoms[(nr % mol.n_atoms) as usize];
            atom_type = atom.atom_type.clone();
        }
        atom_type
    }

    /// Add an existing [`Molecule`] to [`Self`]
    pub fn molecule_existing_add(&mut self, mut molecule: Molecule) {
        molecule.id = self.molecules.last().map(|mol| mol.id + 1).unwrap_or(1);
        self.molecules.push(molecule);
        self.molecule_cnt_list.push(0);
        self.n_molecules += 1;
    }

    /// Set the count of a molecule
    pub fn set_molecule_cnt(&mut self, molecule_idx: usize, count: i64) {
        let old_count;
        if !self.var_num_atoms {
            old_count = self.molecule_cnt_list[molecule_idx];
            self.molecule_cnt_list[molecule_idx] = count;

            self.n_particles += (count - old_count) * self.molecules[molecule_idx].n_atoms;
        } else {
            old_count = self.current_trajectory_frame_set.molecule_cnt_list[molecule_idx];
            self.current_trajectory_frame_set.molecule_cnt_list[molecule_idx] = count;

            self.current_trajectory_frame_set.n_particles +=
                (count - old_count) * self.molecules[molecule_idx].n_atoms;
        }
    }

    /// Get the count of a molecule.
    pub fn get_molecule_cnt(&self, molecule_idx: usize) -> usize {
        usize::try_from(self.molecule_cnt_list[molecule_idx]).expect("usize from i64")
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

        let mut from_atoms = Vec::with_capacity(n_bonds);
        let mut to_atoms = Vec::with_capacity(n_bonds);

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

    /// Retrieve non-particle data, from the last read frame set.
    /// Returns (values, n_frames, n_values_per_frame, data_type).
    pub fn data_get(&mut self, block_id: BlockID) -> Option<(Vec<f64>, i64, i64, DataType)> {
        let (n_frames, _n_particles, n_values_per_frame, data_type, values) =
            self.gen_data_vector_get(false, block_id)?;
        Some((values, n_frames, n_values_per_frame, data_type))
    }

    /// Retrieve particle data, from the last read frame set.
    /// Returns (values, n_frames, n_particles, n_values_per_frame, data_type).
    pub fn particle_data_get(
        &mut self,
        block_id: BlockID,
    ) -> Option<(Vec<f64>, i64, i64, i64, DataType)> {
        let (n_frames, n_particles, n_values_per_frame, data_type, values) =
            self.gen_data_vector_get(true, block_id)?;
        Some((values, n_frames, n_particles, n_values_per_frame, data_type))
    }

    /// Internal: retrieve a vector (1D array) of data from the last read frame set
    fn gen_data_vector_get(
        &mut self,
        is_particle_data: bool,
        block_id: BlockID,
    ) -> Option<(i64, i64, i64, DataType, Vec<f64>)> {
        let mut n_particles = 0;
        let mut block_index = -1;

        let data = if is_particle_data {
            self.particle_data_find(block_id)
        } else {
            self.data_find(block_id)
        };

        if data.is_none() {
            let mut block = GenBlock::new();
            let mut file_pos = self.get_input_file_position();

            // Read all blocks until next frame set block
            self.block_header_read(&mut block);
            loop {
                if file_pos >= self.input_file_len {
                    break;
                }
                match block.id {
                    BlockID::Unknown | BlockID::TrajectoryFrameSet => break,
                    _ => {}
                }

                // Use hash by default (also TODO)
                self.block_read_next(&mut block);
                file_pos = self.get_input_file_position();
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
        Some((
            n_frames,
            n_particles,
            n_values_per_frame,
            data_type,
            float_values,
        ))
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
        let mut file_pos = self.get_input_file_position();
        let mut block = GenBlock::new();

        // Read block headers first to see what block is found
        self.block_header_read(&mut block);

        if block.id != BlockID::TrajectoryFrameSet || block.id == BlockID::Unknown {
            return Err(());
        }

        self.current_trajectory_frame_set_input_file_pos =
            i64::try_from(file_pos).expect("u64 to i64");

        // TODO: make this fallible?
        self.block_read_next(&mut block);
        if block.id != BlockID::Unknown {
            self.n_trajectory_frame_sets += 1;
            file_pos = self.get_input_file_position();

            // Read all blocks until next frame set block
            self.block_header_read(&mut block);
            loop {
                if file_pos >= self.input_file_len {
                    break;
                }
                match block.id {
                    BlockID::Unknown | BlockID::TrajectoryFrameSet => break,
                    _ => {}
                }
                self.block_read_next(&mut block);
                file_pos = self.get_input_file_position();
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

    /// Write one frame set, including mapping and related data blocks to [`self.output_file`]
    /// of [`Self`]
    pub fn frame_set_write(&mut self, hash_mode: bool) -> Result<(), TngError> {
        let frame_set = &self.current_trajectory_frame_set;
        if frame_set.n_written_frames == frame_set.n_frames {
            return Ok(());
        }

        self.current_trajectory_frame_set_output_file_pos =
            i64::try_from(self.get_output_file_position()).expect("i64 from u64");
        self.last_trajectory_frame_set_output_pos =
            self.current_trajectory_frame_set_output_file_pos;

        if self.current_trajectory_frame_set_output_file_pos <= 0 {
            return Err(TngError::Constraint(
                "output file position is invalid; cannot write frame set".to_string(),
            ));
        }

        if self.first_trajectory_frame_set_output_pos == -1 {
            self.first_trajectory_frame_set_output_pos =
                self.current_trajectory_frame_set_output_file_pos;
        }

        let mut block = GenBlock::new();

        self.frame_set_block_write(&mut block);

        // Write non-particle data blocks
        for i in 0..self.current_trajectory_frame_set.n_data_blocks {
            block.id = self.current_trajectory_frame_set.tr_data[i].block_id;
            self.data_block_write(&mut block, i, false, &None, hash_mode);
        }

        // Write the mapping blocks and particle data blocks
        if self.current_trajectory_frame_set.n_mapping_blocks > 0 {
            for i in 0..usize::try_from(self.current_trajectory_frame_set.n_mapping_blocks)
                .expect("usize from i64")
            {
                block.id = BlockID::ParticleMapping;
                if self.current_trajectory_frame_set.mappings[i].n_particles > 0 {
                    self.trajectory_mapping_block_write(&mut block, i, hash_mode);
                    for j in 0..self.current_trajectory_frame_set.n_particle_data_blocks {
                        block.id = self.current_trajectory_frame_set.tr_particle_data[i].block_id;
                        self.data_block_write(
                            &mut block,
                            j,
                            true,
                            &Some(self.current_trajectory_frame_set.mappings[i].clone()),
                            hash_mode,
                        );
                    }
                }
            }
        } else {
            for i in 0..self.current_trajectory_frame_set.n_particle_data_blocks {
                block.id = self.current_trajectory_frame_set.tr_particle_data[i].block_id;
                self.data_block_write(&mut block, i, true, &None, hash_mode);
            }
        }

        // Update pointers in the general info block
        let result = self.header_pointers_update(hash_mode);

        if result.is_ok() {
            self.frame_set_pointers_update(hash_mode);
        }

        self.current_trajectory_frame_set.n_unwritten_frames = 0;

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
        file_pos = i64::try_from(self.get_input_file_position()).expect("i64 from u64");

        found_flag = true;

        // Read only blocks of the request ID until next frame set block
        self.block_header_read(&mut block);
        while file_pos < i64::try_from(self.input_file_len).expect("i64 from u64")
            && block.id != BlockID::TrajectoryFrameSet
            && block.id != BlockID::Unknown
        {
            if block.id == match_block_id {
                self.block_read_next(&mut block);
                file_pos = i64::try_from(self.get_input_file_position()).expect("i64 from u64");
                found_flag = true;
                if file_pos < i64::try_from(self.input_file_len).expect("i64 from u64") {
                    self.block_header_read(&mut block);
                }
            } else {
                file_pos += i64::try_from(block.block_contents_size + block.header_contents_size)
                    .expect("i64 from u64");
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Current(
                        i64::try_from(block.block_contents_size).expect("i64 from u64"),
                    ))
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

    /// Get the dependency of a data block of a specific ID.
    pub fn data_block_dependency_get(&mut self, match_block_id: BlockID) -> Result<u8, ()> {
        for i in 0..self.n_particle_data_blocks {
            let data = &self.non_tr_particle_data[i];
            if data.block_id == match_block_id {
                return Ok(PARTICLE_DEPENDENT);
            }
        }

        for i in 0..self.n_data_blocks {
            let data = &self.non_tr_data[i];
            if data.block_id == match_block_id {
                return Ok(0);
            }
        }

        let particle_data_result = self.particle_data_find(match_block_id);
        if particle_data_result.is_some() {
            return Ok(PARTICLE_DEPENDENT + FRAME_DEPENDENT);
        } else {
            let data_result = self.data_find(match_block_id);
            if data_result.is_some() {
                return Ok(FRAME_DEPENDENT);
            } else {
                let result = self.frame_set_read_current_only_data_from_block_id(match_block_id);
                if result.is_err() {
                    return Err(());
                }
                let particle_data_result = self.particle_data_find(match_block_id);
                if particle_data_result.is_some() {
                    return Ok(PARTICLE_DEPENDENT + FRAME_DEPENDENT);
                } else {
                    let data_result = self.data_find(match_block_id);
                    if data_result.is_some() {
                        return Ok(FRAME_DEPENDENT);
                    }
                }
            }
        }

        Err(())
    }

    /// Get the number of values per frame of a data block of a specific ID.
    pub fn data_block_num_values_per_frame_get(
        &mut self,
        match_block_id: BlockID,
    ) -> Result<i64, ()> {
        for i in 0..self.n_particle_data_blocks {
            let data = &self.non_tr_particle_data[i];
            if data.block_id == match_block_id {
                return Ok(data.n_values_per_frame);
            }
        }

        for i in 0..self.n_data_blocks {
            let data = &self.non_tr_data[i];
            if data.block_id == match_block_id {
                return Ok(data.n_values_per_frame);
            }
        }

        let stat = self.particle_data_find(match_block_id);
        if let Some(data) = stat {
            return Ok(data.n_values_per_frame);
        } else {
            let stat = self.data_find(match_block_id);
            if let Some(data) = stat {
                return Ok(data.n_values_per_frame);
            } else {
                let stat = self.frame_set_read_current_only_data_from_block_id(match_block_id);
                if stat.is_err() {
                    return Err(());
                }
                let stat = self.particle_data_find(match_block_id);
                if let Some(data) = stat {
                    return Ok(data.n_values_per_frame);
                } else {
                    let stat = self.data_find(match_block_id);
                    if let Some(data) = stat {
                        return Ok(data.n_values_per_frame);
                    }
                }
            }
        }
        Err(())
    }

    /// Read the number of the first frame of the next frame set.
    fn first_frame_nr_of_next_frame_set_get(&mut self) -> Option<i64> {
        let file_pos = self.get_input_file_position();

        let next_frame_set_file_pos = if self.current_trajectory_frame_set_input_file_pos <= 0 {
            self.first_trajectory_frame_set_input_pos
        } else {
            self.current_trajectory_frame_set.next_frame_set_file_pos
        };

        if next_frame_set_file_pos <= 0 {
            return None;
        }

        let inp_file = self.input_file.as_mut().expect("init input_file");
        inp_file
            .seek(SeekFrom::Start(
                u64::try_from(next_frame_set_file_pos).expect("u64 from i64"),
            ))
            .expect("no error handling");
        let mut block = GenBlock::new();
        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("Cannot read block header at pos {file_pos}");
        }

        let inp_file = self.input_file.as_mut().expect("init input_file");
        let frame = utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        inp_file
            .seek(SeekFrom::Start(file_pos))
            .expect("no error handling");
        Some(frame)
    }

    pub fn num_frames_get(&mut self) -> Option<i64> {
        let file_pos = self.get_input_file_position();
        let last_file_pos = self.last_trajectory_frame_set_input_pos;

        if last_file_pos <= 0 {
            return None;
        }

        // we manually drop the newly created block as we just want to read
        // the block contents to advanced the reading position
        {
            let mut block = GenBlock::new();
            let inp_file = self.input_file.as_mut().expect("init input_file");
            inp_file
                .seek(SeekFrom::Start(
                    u64::try_from(last_file_pos).expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.block_header_read(&mut block);
            if block.id != BlockID::TrajectoryFrameSet {
                panic!("Cannot read block header at pos {file_pos}");
            };
            drop(block)
        }

        let inp_file = self.input_file.as_mut().expect("init input_file");
        let first_frame = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        let n_frames = utils::read_i64(inp_file, self.endianness64, self.input_swap64);
        inp_file
            .seek(SeekFrom::Start(file_pos))
            .expect("no error handling");

        Some(first_frame + n_frames)
    }

    /// Find the frame set containing a specific frame
    fn frame_set_of_frame_find(&mut self, match_frame: i64) -> Result<(), ()> {
        let mut block = GenBlock::new();
        let mut file_pos = 0;
        if self.current_trajectory_frame_set_input_file_pos < 0 {
            file_pos = self.first_trajectory_frame_set_input_pos;
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.current_trajectory_frame_set_input_file_pos = file_pos;

            // Read block headers first to see what block is found
            self.block_header_read(&mut block);
            if block.id != BlockID::TrajectoryFrameSet {
                panic!("Cannot read block header at pos {file_pos}");
            }

            self.block_read_next(&mut block);
        }

        let frame_set = &self.current_trajectory_frame_set;
        let first_frame = max(frame_set.first_frame, 0);
        let last_frame = first_frame + frame_set.n_frames - 1;

        // Is this the right frame set?
        if first_frame <= match_frame && match_frame <= last_frame {
            return Ok(());
        }

        let mut n_frames_per_frame_set = self.frame_set_n_frames;
        let long_stride_length = self.long_stride_length;
        let medium_stride_length = self.medium_stride_length;

        let temp_frame = self.first_frame_nr_of_next_frame_set_get();
        if let Some(frame) = temp_frame
            && frame - first_frame > n_frames_per_frame_set
        {
            n_frames_per_frame_set = frame - first_frame;
        }

        let n_frames = self.num_frames_get().expect("able to get n_frames");

        if match_frame >= n_frames {
            return Err(());
        }

        if first_frame - match_frame >= match_frame
            || match_frame - last_frame > self.n_trajectory_frame_sets * n_frames_per_frame_set
        {
            // Start from the beginning
            if first_frame - match_frame >= match_frame {
                file_pos = self.first_trajectory_frame_set_input_pos;
                if file_pos <= 0 {
                    return Err(());
                }
            } else if match_frame - first_frame > (n_frames - 1) - match_frame {
                file_pos = self.last_trajectory_frame_set_input_pos;
                // If the last frame set position is not set start from the current
                // frame set, since it will be closer than the first frame set
            }
            // Start from current
            else {
                file_pos = self.current_trajectory_frame_set_input_file_pos;
            }

            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");
                self.current_trajectory_frame_set_input_file_pos = file_pos;

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }
        }

        let mut first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
        let mut last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;

        if match_frame >= first_frame && match_frame <= last_frame {
            return Ok(());
        }

        file_pos = self.current_trajectory_frame_set_input_file_pos;

        // Take long steps forward until a long step forward would be too long or
        // the right frame is found
        while file_pos > 0
            && first_frame + long_stride_length * n_frames_per_frame_set <= match_frame
        {
            file_pos = self
                .current_trajectory_frame_set
                .long_stride_next_frame_set_file_pos;
            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if match_frame >= first_frame && match_frame <= last_frame {
                return Ok(());
            }
        }

        // Take medium steps forward until a medium step forward would be too long or
        // the right frame is found
        while file_pos > 0
            && first_frame + medium_stride_length * n_frames_per_frame_set <= match_frame
        {
            file_pos = self
                .current_trajectory_frame_set
                .medium_stride_next_frame_set_file_pos;
            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if match_frame >= first_frame && match_frame <= last_frame {
                return Ok(());
            }
        }

        // Take one step forward until the right frame is found
        while file_pos > 0 && first_frame < match_frame && last_frame < match_frame {
            file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if match_frame >= first_frame && match_frame <= last_frame {
                return Ok(());
            }
        }

        // Take long steps backward until a long step backward would be too long or
        // the right frame is found
        while file_pos > 0
            && first_frame - long_stride_length * n_frames_per_frame_set >= match_frame
        {
            file_pos = self
                .current_trajectory_frame_set
                .long_stride_prev_frame_set_file_pos;
            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if match_frame >= first_frame && match_frame <= last_frame {
                return Ok(());
            }
        }

        // Take medium steps backward until a medium step backward would be too long or
        // the right frame is found
        while file_pos > 0
            && first_frame - medium_stride_length * n_frames_per_frame_set >= match_frame
        {
            file_pos = self
                .current_trajectory_frame_set
                .medium_stride_prev_frame_set_file_pos;
            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if match_frame >= first_frame && match_frame <= last_frame {
                return Ok(());
            }
        }

        // Take one step backward until the right frame is found
        while file_pos > 0 && first_frame > match_frame && last_frame > match_frame {
            file_pos = self.current_trajectory_frame_set.prev_frame_set_file_pos;
            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if match_frame >= first_frame && match_frame <= last_frame {
                return Ok(());
            }
        }

        // If for some reason the current frame set is not yet found
        // take one step forward until the right frame set is found
        while file_pos > 0 && first_frame < match_frame && last_frame < match_frame {
            file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
            if file_pos > 0 {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(file_pos).expect("u64 from i64"),
                    ))
                    .expect("no error handling");

                // Read block headers first to see what block is found
                self.block_header_read(&mut block);
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if match_frame >= first_frame && match_frame <= last_frame {
                return Ok(());
            }
        }

        Err(())
    }

    /// Update the frame set pointers in the file header (general info block),
    /// already written to disk
    /// `hash_mode` specifies whether to update the block md5 hash when updating the pointers
    fn header_pointers_update(&mut self, _hash_mode: bool) -> Result<(), TngError> {
        self.output_file_init();

        // Save original input_file, replace with a dup of output_file
        let temp = self.input_file.take();
        self.input_file = Some(
            self.output_file
                .as_ref()
                .expect("init output_file")
                .try_clone()
                .expect("dup output file"),
        );

        let mut block = GenBlock::new();

        let output_file = self
            .output_file
            .as_mut()
            .expect("just initialized output file");
        // ftello
        let output_file_pos = output_file.stream_position().expect("no error handling");
        // fseeko
        output_file
            .seek(SeekFrom::Start(0))
            .expect("no error handling");

        if self.block_header_read(&mut block).is_err() {
            self.input_file = temp;
            return Err(TngError::Critical(format!(
                "Cannot read general info header."
            )));
        };

        let output_file = self
            .output_file
            .as_mut()
            .expect("just initialized output file");

        // TODO: hash mode
        let contents_start_pos = output_file.stream_position().expect("no error handling");
        output_file
            .seek(SeekFrom::Current(
                i64::try_from(
                    block.block_contents_size
                        - u64::try_from(5 * std::mem::size_of::<i64>()).expect("u64 from usize"),
                )
                .expect("i64 from u64"),
            ))
            .expect("no error handling");

        self.input_file = temp;

        let mut pos = self.first_trajectory_frame_set_output_pos;
        utils::write_u64(
            output_file,
            u64::try_from(pos).expect("u64 from i64"),
            self.endianness64,
            self.input_swap64,
        );

        pos = self.last_trajectory_frame_set_output_pos;
        utils::write_u64(
            output_file,
            u64::try_from(pos).expect("u64 from i64"),
            self.endianness64,
            self.input_swap64,
        );

        // TODO: hash mode
        // tng_io.c line 1282

        output_file.seek(SeekFrom::Start(output_file_pos));

        Ok(())
    }

    fn reread_frame_set_at_file_pos(&mut self, pos: u64) {
        let mut block = GenBlock::new();

        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(pos))
            .expect("no error handling");

        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("Cannot read block header at pos {pos}");
        }

        self.block_read_next(&mut block);
    }

    /// Get the stride length of a specific data (particle dependency does not matter) block,
    /// either in the current frame set or of a specific frame
    pub fn data_get_stride_length(
        &mut self,
        match_block_id: BlockID,
        frame: i64,
    ) -> Result<i64, ()> {
        let mut new_frame = frame;
        if self.current_trajectory_frame_set_input_file_pos <= 0 {
            new_frame = 0;
        }

        let stat = self.frame_set_of_frame_find(new_frame);
        if stat.is_err() {
            return Err(());
        }

        let orig_file_pos =
            u64::try_from(self.current_trajectory_frame_set_input_file_pos).expect("u64 from i64");
        let stat = self.data_find(match_block_id);

        let is_particle_data;
        let data;
        if stat.is_none() {
            let stat = self.particle_data_find(match_block_id);
            if stat.is_none() {
                let mut stat = self.frame_set_read_current_only_data_from_block_id(match_block_id);

                // If no specific frame was required read until this data block is found
                if new_frame < 0 {
                    let mut file_pos = self.get_input_file_position();

                    while stat.is_err() && file_pos < self.input_file_len {
                        stat = self.frame_set_read_next_only_data_from_block_id(match_block_id);
                        file_pos = self.get_input_file_position();
                    }
                }

                if stat.is_err() {
                    self.reread_frame_set_at_file_pos(orig_file_pos);
                    return Err(());
                }

                let mut stat = self.data_find(match_block_id);
                if stat.is_none() {
                    stat = self.particle_data_find(match_block_id);
                    if let Some(stat) = stat {
                        data = stat;
                        is_particle_data = true;
                    } else {
                        self.reread_frame_set_at_file_pos(orig_file_pos);
                        return Err(());
                    }
                } else {
                    data = stat.unwrap();
                    is_particle_data = false;
                }
            } else {
                data = stat.unwrap();
                is_particle_data = true;
            }
        } else {
            data = stat.unwrap();
            is_particle_data = false;
        }

        if is_particle_data {
            return Ok(data.stride_length);
        }
        // From c code:
        // Possible bug here
        //    else
        //    {
        //        *stride_length = data->stride_length;
        //    }
        self.reread_frame_set_at_file_pos(orig_file_pos);

        Err(())
    }

    /// Add a molecule to the trajectory.
    /// Returns the index of the new molecule in `self.molecules`.
    pub fn add_molecule(&mut self, name: &str) -> usize {
        let id = self.molecules.last().map(|m| m.id + 1).unwrap_or(1);
        self.add_molecule_w_id(name, id)
    }

    fn add_molecule_w_id(&mut self, name: &str, id: i64) -> usize {
        let mut molecule = Molecule::new();
        let length = name.floor_char_boundary(MAX_STR_LEN - 1);
        molecule.name = name[..length].to_string();
        molecule.id = id;

        let idx = self.molecules.len();
        self.molecules.push(molecule);
        self.molecule_cnt_list.push(0);
        self.n_molecules += 1;
        idx
    }

    /// Add a chain to the molecule at `molecule_idx`.
    /// Returns the index of the new chain in `self.molecules[molecule_idx].chains`.
    pub fn add_chain(&mut self, molecule_idx: usize, name: &str) -> usize {
        let id = self.molecules[molecule_idx]
            .chains
            .last()
            .map(|c| c.id + 1)
            .unwrap_or(1);
        self.add_chain_w_id(molecule_idx, name, id)
    }

    fn add_chain_w_id(&mut self, molecule_idx: usize, name: &str, id: u64) -> usize {
        let mut chain = Chain::new();
        chain.set_name(name.to_string());
        chain.parent_molecule_idx = molecule_idx;
        chain.id = id;
        chain.n_residues = 0;

        let mol = &mut self.molecules[molecule_idx];
        chain.residues_indices = (mol.residues.len(), mol.residues.len());
        let idx = mol.chains.len();
        mol.chains.push(chain);
        mol.n_chains += 1;
        idx
    }

    /// Add a residue to the molecule at `molecule_idx`, associated with chain `chain_idx`.
    /// Returns the index of the new residue in `self.molecules[molecule_idx].residues`.
    pub fn add_chain_residue(
        &mut self,
        molecule_idx: usize,
        chain_idx: usize,
        name: &str,
    ) -> usize {
        let mol = &self.molecules[molecule_idx];
        let chain = &mol.chains[chain_idx];
        let id = if chain.n_residues > 0 {
            let last_res_idx = chain.residues_indices.1 - 1;
            mol.residues[last_res_idx].id + 1
        } else {
            0
        };
        self.add_residue_w_id(molecule_idx, chain_idx, name, id)
    }

    fn add_residue_w_id(
        &mut self,
        molecule_idx: usize,
        chain_idx: usize,
        name: &str,
        id: u64,
    ) -> usize {
        let mol = &mut self.molecules[molecule_idx];

        let insert_pos = mol.chains[chain_idx].residues_indices.1;

        let mut residue = Residue::new();
        let length = name.floor_char_boundary(MAX_STR_LEN - 1);
        residue.name = name[..length].to_string();
        residue.chain_index = Some(chain_idx);
        residue.id = id;
        residue.n_atoms = 0;
        residue.atoms_offset = mol.atoms.len();

        mol.residues.insert(insert_pos, residue);

        mol.chains[chain_idx].residues_indices.1 += 1;
        mol.chains[chain_idx].n_residues += 1;

        for c in &mut mol.chains[(chain_idx + 1)..] {
            c.residues_indices.0 += 1;
            c.residues_indices.1 += 1;
        }

        for atom in &mut mol.atoms {
            if let Some(ri) = atom.residue_index {
                if ri >= insert_pos {
                    atom.residue_index = Some(ri + 1);
                }
            }
        }

        mol.n_residues += 1;
        insert_pos
    }

    /// Add an atom to the molecule at `molecule_idx`, associated with residue `residue_idx`.
    /// Returns the index of the new atom in `self.molecules[molecule_idx].atoms`.
    pub fn add_residue_atom(
        &mut self,
        molecule_idx: usize,
        residue_idx: usize,
        name: &str,
        atom_type: &str,
    ) -> usize {
        let id = self.molecules[molecule_idx]
            .atoms
            .last()
            .map(|a| a.id + 1)
            .unwrap_or(0);
        self.add_atom_w_id(molecule_idx, residue_idx, name, atom_type, id)
    }

    fn add_atom_w_id(
        &mut self,
        molecule_idx: usize,
        residue_idx: usize,
        name: &str,
        atom_type: &str,
        id: i64,
    ) -> usize {
        let mol = &mut self.molecules[molecule_idx];

        if mol.residues[residue_idx].n_atoms == 0 {
            mol.residues[residue_idx].atoms_offset = mol.atoms.len();
        }

        let mut atom = Atom::new();
        let len = name.floor_char_boundary(MAX_STR_LEN - 1);
        atom.name = name[..len].to_string();
        let len = atom_type.floor_char_boundary(MAX_STR_LEN - 1);
        atom.atom_type = atom_type[..len].to_string();
        atom.residue_index = Some(residue_idx);
        atom.id = id;

        let idx = mol.atoms.len();
        mol.atoms.push(atom);
        mol.residues[residue_idx].n_atoms += 1;
        mol.n_atoms += 1;
        idx
    }

    /// Add a bond to the molecule at `molecule_idx`.
    /// Returns the index of the new bond in `self.molecules[molecule_idx].bonds`.
    pub fn add_molecule_bond(
        &mut self,
        molecule_idx: usize,
        from_atom_id: i64,
        to_atom_id: i64,
    ) -> usize {
        let mol = &mut self.molecules[molecule_idx];
        let mut bond = Bond::new();
        bond.from_atom_id = from_atom_id;
        bond.to_atom_id = to_atom_id;
        let idx = mol.bonds.len();
        mol.bonds.push(bond);
        mol.n_bonds += 1;
        idx
    }

    pub fn add_data_block(
        &mut self,
        id: BlockID,
        block_name: &str,
        data_type: DataType,
        block_type_flag: bool,
        n_frames: i64,
        n_values_per_frame: i64,
        stride_length: i64,
        codec_id: Compression,
        new_data: Option<Vec<u8>>,
    ) -> Result<(), TngError> {
        if n_values_per_frame <= 0 {
            return Err(TngError::Constraint(format!(
                "`n_values_per_frame` must be a positive integer. Got {n_values_per_frame}"
            )));
        }

        self.add_gen_data_block(
            id,
            false,
            block_name,
            data_type,
            block_type_flag,
            n_frames,
            n_values_per_frame,
            stride_length,
            0,
            0,
            codec_id,
            new_data,
        );
        Ok(())
    }

    fn add_gen_data_block(
        &mut self,
        id: BlockID,
        is_particle_data: bool,
        block_name: &str,
        data_type: DataType,
        block_type_flag: bool,
        n_frames: i64,
        n_values_per_frame: i64,
        stride_length: i64,
        num_first_particle: u64,
        n_particles: i64,
        codec_id: Compression,
        new_data: Option<Vec<u8>>,
    ) {
        let mut stride_length = stride_length;
        if stride_length <= 0 {
            stride_length = 0;
        }

        let found_block = if is_particle_data {
            self.particle_data_find(id)
        } else {
            self.data_find(id)
        };

        let mut data = Data::default();
        // If the block does not exist, create it
        if found_block.is_none() {
            if is_particle_data {
                self.particle_data_block_create(block_type_flag);
            } else {
                self.data_block_create(block_type_flag);
            }

            let frame_set = &self.current_trajectory_frame_set;
            if is_particle_data {
                // block_type_flag == true corresponds to TNG_TRAJECTORY_BLOCK
                data = if block_type_flag {
                    frame_set.tr_particle_data[frame_set.n_particle_data_blocks - 1].clone()
                } else {
                    self.non_tr_particle_data[self.n_particle_data_blocks - 1].clone()
                };
            } else {
                data = if block_type_flag {
                    frame_set.tr_data[frame_set.n_data_blocks - 1].clone()
                } else {
                    self.non_tr_data[self.n_data_blocks - 1].clone()
                };
            }
            data.block_id = id;
            let length = block_name.floor_char_boundary(MAX_STR_LEN - 1);
            data.block_name = block_name[..length].to_string();
            data.values = None;
            data.strings = None;
            data.last_retrieved_frame = -1;
        }
        data.data_type = data_type;
        data.stride_length = stride_length.max(1);
        data.n_values_per_frame = n_values_per_frame;
        data.n_frames = n_frames;

        if is_particle_data {
            data.dependency = PARTICLE_DEPENDENT;
        } else {
            data.dependency = 0;
        }

        let frame_set = &self.current_trajectory_frame_set;
        if block_type_flag && (n_frames > 1 || frame_set.n_frames == n_frames || stride_length > 1)
        {
            data.dependency = FRAME_DEPENDENT;
        }
        data.codec_id = codec_id;
        data.compression_multiplier = 1.0;
        // FIXME(from C code): this can cause problems
        data.first_frame_with_data = frame_set.first_frame;

        let mut tot_n_particles = 0;
        if is_particle_data {
            tot_n_particles = if block_type_flag && self.var_num_atoms {
                frame_set.n_particles
            } else {
                self.n_particles
            };
        }

        // If data values are supplied add that data to the data block
        if let Some(ref new_data) = new_data {
            // Allocate memory
            if is_particle_data {
                data.allocate_particle_data_mem(
                    n_frames,
                    stride_length,
                    tot_n_particles,
                    n_values_per_frame,
                );
            } else {
                data.allocate_data_mem(n_frames, stride_length, n_values_per_frame);
            }

            if n_frames > frame_set.n_unwritten_frames {
                self.current_trajectory_frame_set.n_unwritten_frames = n_frames;
            }

            let n_frames_div = (n_frames - 1) / stride_length + 1;

            match data_type {
                DataType::Char => {
                    let mut cursor = 0;

                    let strings_3d = match data.strings {
                        Some(ref mut s) => s,
                        None => unreachable!("data.strings was None"),
                    };

                    if is_particle_data {
                        for i in 0..n_frames_div {
                            let first_dim_values = &mut strings_3d[i as usize];
                            for j in num_first_particle
                                ..num_first_particle
                                    + u64::try_from(n_particles).expect("u64 from i64")
                            {
                                let second_dim_values = &mut first_dim_values[j as usize];
                                for k in 0..n_values_per_frame {
                                    let remaining = &new_data[cursor..];
                                    let nul_pos = remaining
                                        .iter()
                                        .position(|&b| b == 0)
                                        .unwrap_or(remaining.len());
                                    let len = (nul_pos + 1).min(MAX_STR_LEN);
                                    let str_bytes = &new_data[cursor..cursor + len - 1];
                                    second_dim_values[k as usize] =
                                        String::from_utf8_lossy(str_bytes).into_owned();
                                    cursor += len;
                                }
                            }
                        }
                    } else {
                        for i in 0..n_frames_div {
                            let second_dim_values = &mut strings_3d[0][i as usize];
                            for j in 0..n_values_per_frame {
                                let remaining = &new_data[cursor..];
                                let nul_pos = remaining
                                    .iter()
                                    .position(|&b| b == 0)
                                    .unwrap_or(remaining.len());
                                let len = (nul_pos + 1).min(MAX_STR_LEN);
                                let str_bytes = &new_data[cursor..cursor + len - 1];
                                second_dim_values[j as usize] =
                                    String::from_utf8_lossy(str_bytes).into_owned();
                                cursor += len;
                            }
                        }
                    }
                }
                _ => {
                    let size = data_type.get_size();
                    let copy_len = if is_particle_data {
                        size * (n_frames_div as usize)
                            * (n_particles as usize)
                            * (n_values_per_frame as usize)
                    } else {
                        size * (n_frames_div as usize) * (n_values_per_frame as usize)
                    };
                    data.values.as_mut().expect("data.values to be Some")[..copy_len]
                        .copy_from_slice(&new_data[..copy_len]);
                }
            }
        }
    }

    fn get_molecule_cnt_list(&self) -> &[i64] {
        if self.var_num_atoms {
            &self.current_trajectory_frame_set.molecule_cnt_list
        } else {
            &self.molecule_cnt_list
        }
    }

    pub(crate) fn particle_data_block_add(
        &mut self,
        id: BlockID,
        block_name: &str,
        data_type: DataType,
        block_type_flag: bool,
        n_frames: i64,
        n_values_per_frame: i64,
        stride_length: i64,
        num_first_particle: u64,
        n_particles: i64,
        codec_id: Compression,
        new_data: Option<Vec<u8>>,
    ) -> Result<(), TngError> {
        if n_values_per_frame <= 0 {
            return Err(TngError::Constraint(format!(
                "`n_values_per_frame` must be a positive integer. Got {n_values_per_frame}"
            )));
        }

        if n_particles < 0 {
            return Err(TngError::Constraint(format!(
                "`n_particles` must be >= 0. Got {n_particles}"
            )));
        }

        self.add_gen_data_block(
            id,
            true,
            block_name,
            data_type,
            block_type_flag,
            n_frames,
            n_values_per_frame,
            stride_length,
            num_first_particle,
            n_particles,
            codec_id,
            new_data,
        );

        Ok(())
    }

    pub(crate) fn get_num_molecules(&self) -> usize {
        let cnt_list = self.get_molecule_cnt_list();

        if cnt_list.is_empty() {
            unimplemented!("error handling");
        }

        cnt_list
            .iter()
            .take(usize::try_from(self.n_molecules).expect("usize from i64"))
            .map(|&x| x as usize)
            .sum()
    }
}
