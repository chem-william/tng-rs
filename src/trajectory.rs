use log::{debug, error, warn};
use std::cmp::{max, min};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::TngError;

use crate::atom::Atom;
use crate::bond::Bond;
use crate::chain::Chain;
use crate::coder::Coder;
use crate::compress::{
    Float, MAGIC_INT_POS, MAGIC_INT_VEL, TNG_COMPRESS_ALGO_POS_BWLZH_INTER,
    TNG_COMPRESS_ALGO_POS_BWLZH_INTRA, TNG_COMPRESS_ALGO_POS_STOPBIT_INTER,
    TNG_COMPRESS_ALGO_POS_TRIPLET_INTER, TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA,
    TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE, TNG_COMPRESS_ALGO_POS_XTC2, TNG_COMPRESS_ALGO_POS_XTC3,
    TNG_COMPRESS_ALGO_VEL_BWLZH_INTER, TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE,
    TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER, TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE,
    TNG_COMPRESS_ALGO_VEL_TRIPLET_INTER, TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE, readbufferfix,
    tng_compress_pos, tng_compress_vel, unquantize, unquantize_inter_differences,
    unquantize_inter_differences_int, unquantize_intra_differences,
    unquantize_intra_differences_first_frame, unquantize_intra_differences_int,
};
use crate::data::{Compression, Data, DataType};
use crate::fix_point::{FixT, fixt_pair_to_f64};
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
const SKIP_HASH: bool = false;

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

#[derive(Debug, PartialEq)]
enum ParticleDependency {
    NonParticleBlockData,
    ParticleBlockData,
}

#[derive(Debug, PartialEq)]
pub(crate) enum BlockType {
    NonTrajectory,
    Trajectory,
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
    pub input_file_path: Option<PathBuf>,
    /// Open handle to the input file (None until opened).
    pub input_file: Option<File>,
    /// Length (in bytes) of the input file.
    pub input_file_len: u64,

    /// Path to the output trajectory file (if any).
    pub output_file_path: Option<PathBuf>,
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
    pub(crate) molecules: Vec<Molecule>,
    /// Count of each molecule type (length = `n_molecules`).
    pub(crate) molecule_cnt_list: Vec<i64>,
    /// Total number of particles (or atoms). If variable, updated per frame set.
    pub n_particles: i64,

    /// File‐offset (in bytes) of the first trajectory frame set in the input.
    pub first_trajectory_frame_set_input_file_pos: i64,
    /// File‐offset (in bytes) of the first trajectory frame set in the output.
    pub first_trajectory_frame_set_output_file_pos: i64,
    /// File‐offset (in bytes) of the last trajectory frame set in the input.
    pub last_trajectory_frame_set_input_file_pos: i64,
    /// File‐offset (in bytes) of the last trajectory frame set in the output.
    pub last_trajectory_frame_set_output_file_pos: i64,

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

    /// C API: `tng_trajectory_init`.
    pub fn new() -> Self {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("able to get time since UNIX_EPOCH")
            .as_secs();

        let (endianness32, endianness64) = Trajectory::detect_host_endianness();

        Trajectory {
            input_file_path: None,
            input_file: None,
            input_file_len: 0,

            output_file_path: None,
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

            first_trajectory_frame_set_input_file_pos: 0,
            first_trajectory_frame_set_output_file_pos: 0,
            last_trajectory_frame_set_input_file_pos: 0,
            last_trajectory_frame_set_output_file_pos: 0,

            current_trajectory_frame_set: TrajectoryFrameSet::new(),
            current_trajectory_frame_set_input_file_pos: -1,
            current_trajectory_frame_set_output_file_pos: -1,
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

    /// C API: `tng_output_file_set`.
    ///
    /// Set the name of the output file.
    pub fn output_file_set(&mut self, path: &Path) {
        if let Some(output_file_path) = self.output_file_path.as_ref()
            && output_file_path == path
        {
            return;
        }

        // If a file was already open, drop (close) it.
        self.output_file.take();

        let truncated = if path.to_str().expect("valid unicode path").len() + 1 > MAX_STR_LEN {
            &path.to_str().unwrap()[..MAX_STR_LEN - 1]
        } else {
            path.to_str().unwrap()
        };

        self.output_file_path = Some(PathBuf::from(truncated.to_string()));

        self.output_file_init();
    }

    pub(crate) fn output_append_file_set(&mut self, filename: &Path) -> Result<(), TngError> {
        if let Some(output_file_path) = self.output_file_path.as_ref()
            && output_file_path == filename
        {
            return Ok(());
        }

        // If a file was already open, drop (close) it.
        self.output_file.take();

        let truncated = if filename.to_str().expect("valid unicode path").len() + 1 > MAX_STR_LEN {
            &filename.to_str().unwrap()[..MAX_STR_LEN - 1]
        } else {
            filename.to_str().unwrap()
        };
        self.output_file_path = Some(PathBuf::from(truncated.to_string()));

        match File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(
                self.output_file_path
                    .as_ref()
                    .expect("we just created output_file_path"),
            ) {
            Ok(f) => {
                self.output_file = Some(f);
            }
            Err(_) => {
                eprintln!(
                    "Cannot open file {}. {}:{}",
                    self.output_file_path
                        .as_ref()
                        .expect("we just created output_file_path")
                        .display(),
                    file!(),
                    line!()
                );
                panic!();
            }
        }

        self.input_file = Some(
            self.output_file
                .as_ref()
                .expect("we just created the output_file")
                .try_clone()?,
        );

        Ok(())
    }

    /// Open the output file is it is not already opened. If the file does not
    /// already exist, create it.
    pub fn output_file_init(&mut self) {
        if self.output_file.is_none() {
            // If no path has ever been set, error out
            if self.output_file_path.is_none() {
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
                .open(path.as_ref().expect("we just checked that it was not None"))
            {
                Ok(f) => {
                    self.output_file = Some(f);
                }
                Err(_) => {
                    eprintln!(
                        "Cannot open file {}. {}:{}",
                        path.expect("we just checked that it was not None")
                            .display(),
                        file!(),
                        line!()
                    );
                    panic!();
                }
            }
        }
    }

    /// C API: `tng_medium_stride_length_set`.
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

    /// C API: `tng_long_stride_length_set`.
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

    /// C API: `tng_first_computer_name_set`.
    pub fn set_first_computer_name(&mut self, new_name: &str) {
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.first_computer_name = new_name[..length].to_string();
    }

    /// C API: `tng_first_program_name_set`.
    pub fn set_first_program_name(&mut self, new_name: &str) {
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.first_program_name = new_name[..length].to_string();
    }

    /// C API: `tng_forcefield_name_set`.
    pub fn set_forcefield_name(&mut self, new_name: &str) {
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.forcefield_name = new_name[..length].to_string();
    }

    /// C API: `tng_time_per_frame_set`.
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

    /// C API: `tng_num_particles_get`.
    pub fn num_particles_get(&self) -> i64 {
        if !self.var_num_atoms {
            self.n_particles
        } else {
            self.current_trajectory_frame_set.n_particles
        }
    }

    /// C API: `tng_input_file_set`.
    ///
    /// Set the name of the input file.
    pub fn input_file_set(&mut self, path: &Path) {
        if let Some(input_file_path) = self.input_file_path.as_ref()
            && input_file_path == path
        {
            return;
        }

        // If a file was already open, drop (close) it.
        self.input_file.take();

        let truncated = if path.to_str().expect("valid unicode path").len() + 1 > MAX_STR_LEN {
            &path.to_str().unwrap()[..MAX_STR_LEN - 1]
        } else {
            path.to_str().unwrap()
        };

        self.input_file_path = Some(PathBuf::from(truncated.to_string()));

        self.input_file_init();
    }

    /// Open the input file if it is not already opened.
    pub fn input_file_init(&mut self) {
        if self.input_file.is_none() {
            // If no path has been set, error out
            if self.input_file_path.is_none() {
                eprintln!("No file specified for reading. {}:{}", file!(), line!());
                panic!();
            }

            // Try to open the file in "rb" mode (read‐only, binary)
            let path = self.input_file_path.clone();
            match File::open(path.as_ref().expect("we just checked it was not None")) {
                Ok(f) => {
                    self.input_file = Some(f);
                }
                Err(_) => {
                    eprintln!(
                        "Cannot open file {}. {}:{}",
                        path.expect("we just checked that it was not None")
                            .display(),
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
                    self.input_swap32 = Some(utils::swap_byte_order_big_endian_32);
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

        let new_pos: u64 = (start_pos as i128 + block.header_contents_size as i128)
            .try_into()
            .expect("set new position when reading block header");

        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(new_pos))
            .map_err(|e| {
                TngError::Critical(format!(
                    "Cannot seek to position {new_pos} in block_header_read: {e}"
                ))
            })?;
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

        self.frame_set_particle_mapping_free();

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

        // If the byte order needs to be swapped the data must be read one value at
        // a time and swapped. Otherwise the data can be read all at once.
        if self.input_swap64.is_some() {
            let inp_file = self.input_file.as_mut().expect("init input_file");
            for i in 0..mapping.n_particles as usize {
                mapping.real_particle_numbers[i] =
                    utils::read_i64(inp_file, self.endianness64, self.input_swap64);
            }
        } else {
            let bytes_to_read = usize::try_from(mapping.n_particles).expect("i64 to usize")
                * std::mem::size_of::<i64>();
            let mut buffer = vec![0u8; bytes_to_read];

            let inp_file = self.input_file.as_mut().expect("init input_file");
            match inp_file.read_exact(&mut buffer) {
                Ok(()) => {
                    for (i, chunk) in buffer.chunks_exact(8).enumerate() {
                        let bytes: [u8; 8] = chunk.try_into().expect("chunk is exactly 8 bytes");
                        mapping.real_particle_numbers[i] = i64::from_ne_bytes(bytes);
                    }
                }
                Err(_) => {
                    eprintln!("Cannot read block. {}:{}", file!(), line!());
                    panic!()
                }
            }
        }
        // TODO: Handle hashing
        self.current_trajectory_frame_set.mappings.push(mapping);

        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(start_pos + block.block_contents_size))
            .expect("no error handling");
    }

    /// Write the atom mappings of the current trajectory frame set
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
        let _header_file_pos = self
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
        self.first_trajectory_frame_set_input_file_pos =
            utils::read_i64(inp_file, self.endianness64, self.input_swap64);

        self.current_trajectory_frame_set.next_frame_set_file_pos =
            self.first_trajectory_frame_set_input_file_pos;
        self.last_trajectory_frame_set_input_file_pos =
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

                    residue.parent_molecule_idx = mol_idx as usize;
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
                        atom.parent_molecule_idx = mol_idx as usize;
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

                    residue.parent_molecule_idx = mol_idx as usize;
                    // Link to no chain: `residue->chain = 0;`
                    residue.chain_index = None;
                    residue.name = String::new();

                    residue.read_data(self);

                    residue.atoms_offset = atom_idx;
                    let atom_count = residue.n_atoms;
                    for _ in 0..atom_count {
                        let mut atom = Atom::new();

                        atom.parent_molecule_idx = mol_idx as usize;
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
                    atom.parent_molecule_idx = mol_idx as usize;
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
    fn data_block_meta_information_read(&mut self, _block: &mut GenBlock) -> BlockMetaInfo {
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

    fn particle_data_block_create(&mut self, block_type_flag: &BlockType) {
        let frame_set = &mut self.current_trajectory_frame_set;
        match block_type_flag {
            BlockType::Trajectory => {
                frame_set.n_particle_data_blocks += 1;
                frame_set.tr_particle_data.push(Data::default());
            }
            BlockType::NonTrajectory => {
                self.n_particle_data_blocks += 1;
                self.non_tr_particle_data.push(Data::default());
            }
        }
    }

    fn data_block_create(&mut self, block_type_flag: &BlockType) {
        let frame_set = &mut self.current_trajectory_frame_set;
        match block_type_flag {
            BlockType::Trajectory => {
                frame_set.n_data_blocks += 1;
                frame_set.tr_data.push(Data::default());
            }
            BlockType::NonTrajectory => {
                self.n_data_blocks += 1;
                self.non_tr_data.push(Data::default());
            }
        }
    }

    // We don't bother to estimate the max bound of the compression as Rust has dynamic arrays (Vec)
    // which C doesn't thus needing to pre-allocate the maximum amount.
    fn gzip_compress(data: &[u8], len: usize) -> Result<Vec<u8>, ()> {
        let mut encoder = ZlibEncoder::new(Vec::with_capacity(len), flate2::Compression::default());
        encoder.write_all(data).map_err(|_| ())?;

        encoder.finish().map_err(|_| ())
    }

    fn gzip_uncompress(
        data: &[u8],
        compressed_len: u64,
        uncompressed_len: usize,
    ) -> Result<Vec<u8>, TngError> {
        let cursor = &data[..compressed_len as usize];
        let mut decoder = ZlibDecoder::new(cursor);
        let mut output = Vec::with_capacity(uncompressed_len);
        match decoder.read_to_end(&mut output) {
            Ok(_) => {
                if output.len() != uncompressed_len {
                    return Err(TngError::Constraint(format!(
                        "Expected {} bytes, but uncompressed {} bytes.",
                        uncompressed_len,
                        output.len()
                    )));
                }
                Ok(output)
            }
            Err(e) => Err(TngError::Constraint(format!("{e}"))),
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
        let frame_set_n_frames = self.current_trajectory_frame_set.n_frames;

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

        let is_traj_block = if self.current_trajectory_frame_set_input_file_pos > 0 {
            BlockType::Trajectory
        } else {
            BlockType::NonTrajectory
        };

        // Use position-based lookup so we get a mutable reference into the
        // actual vec, not a clone (the old `particle_data_find`/`data_find`
        // returned clones, so writes to the found Data were lost).
        enum FoundIn {
            TrajVec(usize),
            NonTrajVec(usize),
        }

        let found: Option<FoundIn> = if is_particle_data {
            if is_traj_block == BlockType::Trajectory {
                self.current_trajectory_frame_set
                    .tr_particle_data
                    .iter()
                    .position(|d| d.block_id == block.id)
                    .map(FoundIn::TrajVec)
            } else {
                self.non_tr_particle_data
                    .iter()
                    .position(|d| d.block_id == block.id)
                    .map(FoundIn::NonTrajVec)
            }
        } else if is_traj_block == BlockType::Trajectory {
            self.current_trajectory_frame_set
                .tr_data
                .iter()
                .position(|d| d.block_id == block.id)
                .map(FoundIn::TrajVec)
                .or_else(|| {
                    self.non_tr_data
                        .iter()
                        .position(|d| d.block_id == block.id)
                        .map(FoundIn::NonTrajVec)
                })
        } else {
            self.non_tr_data
                .iter()
                .position(|d| d.block_id == block.id)
                .map(FoundIn::NonTrajVec)
        };

        // If the block does not exist, create it
        let data = if let Some(found_in) = found {
            match found_in {
                FoundIn::TrajVec(idx) => {
                    if is_particle_data {
                        &mut self.current_trajectory_frame_set.tr_particle_data[idx]
                    } else {
                        &mut self.current_trajectory_frame_set.tr_data[idx]
                    }
                }
                FoundIn::NonTrajVec(idx) => {
                    if is_particle_data {
                        &mut self.non_tr_particle_data[idx]
                    } else {
                        &mut self.non_tr_data[idx]
                    }
                }
            }
        } else {
            if is_particle_data {
                self.particle_data_block_create(&is_traj_block);
            } else {
                self.data_block_create(&is_traj_block);
            }

            let frame_set = &mut self.current_trajectory_frame_set;
            let data = if is_particle_data {
                if is_traj_block == BlockType::Trajectory {
                    frame_set
                        .tr_particle_data
                        .last_mut()
                        .expect("available tr_particle_data")
                } else {
                    self.non_tr_particle_data
                        .last_mut()
                        .expect("available element on non_tr_particle_data")
                }
            } else if is_traj_block == BlockType::Trajectory {
                frame_set
                    .tr_data
                    .last_mut()
                    .expect("available element on tr_data")
            } else {
                self.non_tr_data
                    .last_mut()
                    .expect("available element on non_tr_data")
            };
            data.values = None;
            // from c - FIXME: Memory leak from strings
            data.strings = None;
            data.n_frames = 0;

            data
        };

        // These fields must be updated regardless of whether the block is new or existing
        data.block_id = block.id;
        data.block_name = block.name.as_ref().expect("block to have a name").clone();
        data.data_type = meta_info.datatype;
        data.codec_id = meta_info.codec_id;
        data.compression_multiplier = meta_info.multiplier;
        data.stride_length = meta_info.stride_length;
        data.last_retrieved_frame = -1;
        data.dependency = 0;
        if is_particle_data {
            data.dependency |= PARTICLE_DEPENDENT;
        }
        if is_traj_block == BlockType::Trajectory
            && (meta_info.n_frames > 1
                || frame_set_n_frames == meta_info.n_frames
                || meta_info.stride_length > 1)
        {
            data.dependency |= FRAME_DEPENDENT;
        }

        let tot_n_particles = if is_particle_data {
            if is_traj_block == BlockType::Trajectory && self.var_num_atoms {
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
                Compression::TNG => {
                    Trajectory::tng_uncompress(block, &meta_info.datatype, &contents, full_data_len)
                        .unwrap()
                }
                Compression::GZip => {
                    Trajectory::gzip_uncompress(&contents, block_data_len, full_data_len).unwrap()
                }
            };
            (actual_contents, full_data_len)
        } else {
            let mut full_data_len = (n_frames_div as usize)
                .checked_mul(size)
                .and_then(|x| x.checked_mul(meta_info.n_values as usize))
                .unwrap_or(0);
            if is_particle_data {
                full_data_len = full_data_len
                    .checked_mul(meta_info.block_n_particles as usize)
                    .expect("mul of meta_info.block_n_particles");
            }
            contents.truncate(full_data_len);
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
                        ..meta_info.num_first_particle + meta_info.block_n_particles
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

        self.data_read(block, meta_info, remaining_len)
            // TODO
            .expect("error handling");

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
    fn block_read_next(&mut self, block: &mut GenBlock, _hash_mode: bool) {
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

    /// C API: `tng_file_headers_read`.
    pub fn file_headers_read(&mut self, hash_mode: bool) {
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
                self.block_read_next(&mut block, hash_mode);
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
        // codec_id is written as u64, not as Compression enum size
        length += size_of::<u64>();

        if is_particle_data {
            length += size_of_val(&num_first_particle) + size_of_val(&n_particles);
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
        block.id = BlockID::GeneralInfo;
        block.block_contents_size = self.general_info_block_len_calculate();
        let _header_file_pos = 0;
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
            self.first_trajectory_frame_set_output_file_pos,
            self.endianness64,
            self.output_swap64,
        );
        utils::write_i64(
            out_file,
            self.last_trajectory_frame_set_output_file_pos,
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

                        let atom_start = residue.atoms_offset;
                        let atom_end = atom_start + residue.n_atoms as usize;
                        let atom_slice = &molecule.atoms[atom_start..atom_end];
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
                    let atom_start = residue.atoms_offset;
                    let atom_end = atom_start + residue.n_atoms as usize;
                    let atom_slice = &molecule.atoms[atom_start..atom_end];
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
        let _header_file_pos = self
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

    /// .
    ///
    /// # Panics
    ///
    /// Panics if .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    fn tng_compress(
        &self,
        compress_algo_pos: &mut Vec<i32>,
        compress_algo_vel: &mut Vec<i32>,
        block: &GenBlock,
        n_frames: i64,
        n_particles: i64,
        data_type: &DataType,
        data: &[u8],
    ) -> Result<Vec<u8>, ()> {
        let dest;

        let mut algo_find_n_frames = -1;
        if block.id != BlockID::TrajPositions && block.id != BlockID::TrajVelocities {
            eprintln!("Can only compress positions and velocities with the TNG method");
            return Err(());
        }

        if *data_type != DataType::Float && *data_type != DataType::Double {
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
                let mut alt_algo = vec![0; nalgo];

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
                        data.len().is_multiple_of(4),
                        "Float‐branch: data_bytes.len() must be exactly count * 4"
                    );

                    // TODO: maybe just re-interpret these bytes
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_ne_bytes(arr));
                    }

                    tng_compress_pos(
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
                        doubles.push(f64::from_ne_bytes(arr));
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
                if compress_algo_pos.is_empty() {
                    *compress_algo_pos = vec![0; nalgo];
                    compress_algo_pos[0] = alt_algo[0];
                    compress_algo_pos[1] = alt_algo[1];
                    compress_algo_pos[2] = -1;
                    compress_algo_pos[3] = -1;
                }
            // TODO: is it a bug in the original code that it checks twice for the compress_algo_pos?
            } else if compress_algo_pos.is_empty()
                || compress_algo_pos[2] == -1
                || compress_algo_pos[2] == -1
            {
                algo_find_n_frames = if n_frames > 6 { 5 } else { n_frames };

                // If the algorithm parameters are -1 they will be determined during the compression.
                if compress_algo_pos.is_empty() {
                    let nalgo = usize::try_from(tng_compress_nalgo()).expect("usize from u64");
                    *compress_algo_pos = vec![-1; nalgo];
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
                        floats.push(f32::from_ne_bytes(arr));
                    }

                    let mut return_dest = tng_compress_pos(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(algo_find_n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        compress_algo_pos,
                    );
                    if algo_find_n_frames < n_frames {
                        return_dest = tng_compress_pos(
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
                        doubles.push(f64::from_ne_bytes(arr));
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
                        floats.push(f32::from_ne_bytes(arr));
                    }
                    tng_compress_pos(
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
                        doubles.push(f64::from_ne_bytes(arr));
                    }
                    tng_compress_pos(
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
                let mut alt_algo = vec![0; nalgo];

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
                        floats.push(f32::from_ne_bytes(arr));
                    }
                    tng_compress_vel(
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
                        doubles.push(f64::from_ne_bytes(arr));
                    }
                    tng_compress_vel(
                        &doubles,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(n_frames).expect("usize from i64"),
                        d_precision,
                        0,
                        &mut alt_algo,
                    )
                };
                // If there had been no algorithm determined before keep the initial coding
                // and initial coding parameter so that they won't have to be determined again
                if compress_algo_vel.is_empty() {
                    *compress_algo_vel = vec![0; nalgo];
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
                    *compress_algo_vel = vec![-1; nalgo];
                }

                dest = if *data_type == DataType::Float {
                    // TODO: maybe just re-interpret these bytes
                    let mut floats = Vec::new();
                    for chunk in data.chunks_exact(4) {
                        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_ne_bytes(arr));
                    }
                    let mut return_dest = tng_compress_vel(
                        &floats,
                        usize::try_from(n_particles).expect("usize from i64"),
                        usize::try_from(algo_find_n_frames).expect("usize from i64"),
                        f_precision,
                        0,
                        compress_algo_vel,
                    );
                    if algo_find_n_frames < n_frames {
                        return_dest = tng_compress_vel(
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
                        doubles.push(f64::from_ne_bytes(arr));
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
                        floats.push(f32::from_ne_bytes(arr));
                    }
                    tng_compress_vel(
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
                        doubles.push(f64::from_ne_bytes(arr));
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

        dest.ok_or(())
    }

    fn tng_uncompress(
        block: &GenBlock,
        data_type: &DataType,
        data: &[u8],
        uncompressed_len: usize,
    ) -> Result<Vec<u8>, TngError> {
        if block.id != BlockID::TrajPositions && block.id != BlockID::TrajVelocities {
            return Err(TngError::Constraint(
                "Can only uncompress positions and velocities with the TNG method".to_string(),
            ));
        }

        match *data_type {
            DataType::Float => {
                let mut f_dest = vec![0.0f32; uncompressed_len];
                compress_uncompress(data, &mut f_dest)?;
                Ok(f_dest.iter().flat_map(|&f| f.to_ne_bytes()).collect())
            }
            DataType::Double => {
                let mut d_dest = vec![0.0f64; uncompressed_len];
                compress_uncompress(data, &mut d_dest)?;
                Ok(d_dest.iter().flat_map(|&f| f.to_ne_bytes()).collect())
            }
            DataType::Char | DataType::Int => Err(TngError::Constraint(format!(
                "Data type not supported. Got {data_type:?}"
            ))),
        }
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

        let frame_step = (n_frames - 1) / stride_length + 1;

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
                contents[..full_data_len].copy_from_slice(&values[..full_data_len]);

                // If writing TNG compressed data the endianness is taken into account by
                // the compression routines. TNG compressed data is always written as little endian
                if cloned_data.codec_id != Compression::TNG {
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
                            Compression::XTC => {
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
                            Compression::TNG => unreachable!("handled by outer guard"),
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
                            Compression::XTC => {
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
                            Compression::TNG => unreachable!("handled by outer guard"),
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
                        Ok(compressed) => {
                            block_data_len = compressed.len();
                            contents = compressed;
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
                                block,
                                block_index,
                                is_particle_data,
                                mapping,
                                hash_mode,
                            );
                            return;
                        }
                    }
                    self.compress_algo_pos = compress_algo_pos;
                    self.compress_algo_vel = compress_algo_vel;
                }
                Compression::GZip => match Self::gzip_compress(&contents, contents.len()) {
                    Ok(compressed) => {
                        block_data_len = compressed.len();
                        contents = compressed;
                    }
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
                block.block_contents_size = block
                    .block_contents_size
                    .checked_sub(u64::try_from(full_data_len).expect("usize to u64"))
                    .expect("block contents size to include uncompressed data length")
                    + u64::try_from(block_data_len).expect("usize to u64");

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

    /// C API: `tng_file_headers_write`.
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
                self.last_trajectory_frame_set_input_file_pos =
                    self.last_trajectory_frame_set_output_file_pos;
            }

            self.reread_frame_set_at_file_pos(
                u64::try_from(self.last_trajectory_frame_set_input_file_pos).expect("u64 from i64"),
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

        let mut pos = self.first_trajectory_frame_set_input_file_pos;

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

        self.block_read_next(&mut block, SKIP_HASH);

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
            .map(|f| f.try_clone().expect("able to clone file"));

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
            let _contents_start_pos = self.get_output_file_position();

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

            let _contents_start_pos = self.get_output_file_position();

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
            let _contents_start_pos = self.get_output_file_position();

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
                            - u64::try_from(size_of::<i64>() + 2 * size_of::<f64>())
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
        self.input_file = temp_input_file
            .as_ref()
            .map(|f| f.try_clone().expect("able to clone file"));
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

        self.input_file
            .as_mut()
            .expect("init input_file")
            .seek(SeekFrom::Start(0))
            .expect("failed to seek to start of input file");

        let mut block = GenBlock::new();

        // Read through the headers of non-trajectory blocks (they come before the
        // trajectory blocks in the file)
        // TODO: error handling
        while len < self.input_file_len
            && self.block_header_read(&mut block).is_ok()
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

    /// C API: `tng_molecule_find`.
    ///
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

    /// C API: `tng_molecule_cnt_list_get`.
    ///
    /// Get the list of the count of each molecule
    pub fn molecule_cnt_list_get(&self) -> &Vec<i64> {
        if self.var_num_atoms {
            &self.current_trajectory_frame_set.molecule_cnt_list
        } else {
            &self.molecule_cnt_list
        }
    }

    /// C API: `tng_molecule_id_of_particle_nr_get`.
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

    /// C API: `tng_residue_id_of_particle_nr_get`.
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

    /// C API: `tng_global_residue_id_of_particle_nr_get`.
    ///
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

    /// C API: `tng_molecule_name_of_particle_nr_get`.
    ///
    /// Get the molecule name of real particle number (number in mol system)
    pub fn molecule_name_of_particle_nr_get(
        &self,
        nr: i64,
        max_len: usize,
    ) -> Result<&str, TngError> {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = Err(TngError::NotFound(
            "could not find molecule name".to_string(),
        ));
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            name = Self::validate_get_name_len(&mol.name, "molecule name", max_len);
        }
        name
    }

    /// C API: `tng_chain_name_of_particle_nr_get`.
    ///
    /// Get the chain name of real particle number (number in mol system)
    pub fn chain_name_of_particle_nr_get(&self, nr: i64, max_len: usize) -> Result<&str, TngError> {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = Err(TngError::NotFound("could not find chain name".to_string()));
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
            name =
                Self::validate_get_name_len(&mol.chains[*chain_index].name, "chain name", max_len);
        }
        name
    }

    /// C API: `tng_residue_name_of_particle_nr_get`.
    ///
    /// Get the residue name of real particle number (number in mol system).
    pub fn residue_name_of_particle_nr_get(
        &self,
        nr: i64,
        max_len: usize,
    ) -> Result<&str, TngError> {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = Err(TngError::NotFound(
            "could not find residue name".to_string(),
        ));
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            let atom = &mol.atoms[(nr % mol.n_atoms) as usize];
            let residue_index = atom.residue_index.expect("atom in residue");
            name = Self::validate_get_name_len(
                &mol.residues[residue_index].name,
                "residue name",
                max_len,
            );
        }
        name
    }

    /// C API: `tng_atom_name_of_particle_nr_get`.
    ///
    /// Get the atom name of real particle number (number in mol system).
    pub fn atom_name_of_particle_nr_get(&self, nr: i64, max_len: usize) -> Result<&str, TngError> {
        let mut count = 0;
        let molecule_count_list = self.molecule_cnt_list_get();

        let mut name = Err(TngError::NotFound(
            "could not find residue name".to_string(),
        ));
        for (mol, mol_count) in self.molecules.iter().zip(molecule_count_list) {
            if count + mol.n_atoms * mol_count - 1 < nr {
                count += mol.n_atoms * mol_count;
                continue;
            }
            let atom = &mol.atoms[(nr % mol.n_atoms) as usize];
            name = Self::validate_get_name_len(&atom.name, "atom name", max_len);
        }
        name
    }

    /// C API: `tng_atom_type_of_particle_nr_get`.
    ///
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

    /// C API: `tng_molecule_existing_add`.
    ///
    /// Add an existing [`Molecule`] to [`Self`]
    pub(crate) fn molecule_existing_add(&mut self, mut molecule: Molecule) {
        molecule.id = self.molecules.last().map(|mol| mol.id + 1).unwrap_or(1);
        self.molecules.push(molecule);
        self.molecule_cnt_list.push(0);
        self.n_molecules += 1;
    }

    /// C API: `tng_molecule_cnt_set`.
    ///
    /// Set the count of a molecule.
    pub fn molecule_cnt_set(&mut self, molecule_idx: usize, cnt: i64) {
        let old_count;
        if !self.var_num_atoms {
            old_count = self.molecule_cnt_list[molecule_idx];
            self.molecule_cnt_list[molecule_idx] = cnt;

            self.n_particles += (cnt - old_count) * self.molecules[molecule_idx].n_atoms;
        } else {
            old_count = self.current_trajectory_frame_set.molecule_cnt_list[molecule_idx];
            self.current_trajectory_frame_set.molecule_cnt_list[molecule_idx] = cnt;

            self.current_trajectory_frame_set.n_particles +=
                (cnt - old_count) * self.molecules[molecule_idx].n_atoms;
        }
    }

    /// C API: `tng_molecule_cnt_get`.
    ///
    /// Get the count of a molecule.
    pub fn get_molecule_cnt(&self, molecule_idx: usize) -> usize {
        usize::try_from(self.molecule_cnt_list[molecule_idx]).expect("usize from i64")
    }

    /// C API: `tng_molsystem_bonds_get`.
    ///
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
                    let bond = &mol.bonds[k as usize];

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

    /// C API: `tng_data_vector_get`.
    ///
    /// Retrieve non-particle data, from the last read frame set.
    /// Returns (values, n_frames, n_values_per_frame, data_type).
    pub fn data_get(&mut self, block_id: BlockID) -> Option<(Vec<f64>, i64, i64, DataType)> {
        let (n_frames, _n_particles, n_values_per_frame, data_type, values) =
            self.gen_data_vector_get(false, block_id)?;
        Some((values, n_frames, n_values_per_frame, data_type))
    }

    /// C API: `tng_particle_data_vector_get`.
    ///
    /// Retrieve particle data, from the last read frame set.
    /// Returns (values, n_frames, n_particles, n_values_per_frame, data_type).
    pub fn particle_data_get(
        &mut self,
        block_id: BlockID,
    ) -> Option<(Vec<f64>, i64, i64, i64, DataType)> {
        let (n_frames, n_particles, n_values_per_frame, data_type, values) =
            self.gen_data_get(true, block_id)?;
        Some((values, n_frames, n_particles, n_values_per_frame, data_type))
    }

    /// C API: `tng_particle_data_interval_get`
    ///
    /// Returns `(values, n_frames, n_particles, n_values_per_frame, data_type)`.
    pub fn particle_data_interval_get(
        &mut self,
        block_id: BlockID,
        start_frame_nr: i64,
        end_frame_nr: i64,
        hash_mode: bool,
    ) -> Result<(Vec<f64>, i64, i64, i64, DataType), TngError> {
        assert!(
            start_frame_nr <= end_frame_nr,
            "`start_frame_nr` ({start_frame_nr}) must not be higher than `end_frame_nr` ({end_frame_nr})"
        );
        let (n_frames, n_particles, n_values_per_frame, data_type, values) =
            self.gen_data_interval_get(block_id, true, start_frame_nr, end_frame_nr, hash_mode)?;
        Ok((values, n_frames, n_particles, n_values_per_frame, data_type))
    }

    /// C API: `tng_particle_data_vector_interval_get`
    ///
    /// Returns `(values, n_particles, n_values_per_frame, stride_length, data_type)`.
    fn particle_data_vector_interval_get(
        &mut self,
        block_id: BlockID,
        start_frame_nr: i64,
        end_frame_nr: i64,
        hash_mode: bool,
    ) -> Result<(Vec<f64>, i64, i64, i64, DataType), TngError> {
        assert!(
            start_frame_nr <= end_frame_nr,
            "`start_frame_nr`, must be lower or equal to `end_frame_nr`"
        );

        self.gen_data_vector_interval_get(block_id, true, start_frame_nr, end_frame_nr, hash_mode)
    }

    /// C API: `tng_data_interval_get`
    ///
    /// Returns `(values, n_frames, n_values_per_frame, data_type)`.
    pub fn data_interval_get(
        &mut self,
        block_id: BlockID,
        start_frame_nr: i64,
        end_frame_nr: i64,
        hash_mode: bool,
    ) -> Result<(Vec<f64>, i64, i64, DataType), TngError> {
        assert!(
            start_frame_nr <= end_frame_nr,
            "`start_frame_nr` ({start_frame_nr}) must not be higher than `end_frame_nr` ({end_frame_nr})"
        );
        let (n_frames, _n_particles, n_values_per_frame, data_type, values) =
            self.gen_data_interval_get(block_id, false, start_frame_nr, end_frame_nr, hash_mode)?;
        Ok((values, n_frames, n_values_per_frame, data_type))
    }

    fn gen_data_vector_interval_get(
        &mut self,
        block_id: BlockID,
        is_particle_data: bool,
        start_frame_nr: i64,
        end_frame_nr: i64,
        hash_mode: bool,
    ) -> Result<(Vec<f64>, i64, i64, i64, DataType), TngError> {
        let frame_set = &self.current_trajectory_frame_set;
        let first_frame = frame_set.first_frame;

        // Do not re-read the frame set if not necessary
        if self.current_trajectory_frame_set_input_file_pos < 0
            || start_frame_nr < first_frame
            || start_frame_nr >= first_frame + self.current_trajectory_frame_set.n_frames
        {
            self.frame_set_of_frame_find(start_frame_nr)?;
        }

        // Re-read the relevant data block
        self.frame_set_read_current_only_data_from_block_id(USE_HASH, block_id)?;

        // (From C) TODO: Test that blocks are read correctly
        let data = if is_particle_data {
            self.particle_data_find(block_id)
        } else {
            self.data_find(block_id)
        };

        if first_frame != self.current_trajectory_frame_set.first_frame || data.is_none() {
            let mut block = GenBlock::new();
            if data.is_none() {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(
                        u64::try_from(self.current_trajectory_frame_set_input_file_pos)
                            .expect("i64 to u64"),
                    ))?;
                self.block_header_read(&mut block)?;
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Current(block.block_contents_size as i64))?;
            }
            let mut file_pos = self.get_input_file_position();
            // Read until next frame set block
            while file_pos < self.input_file_len
                && self.block_header_read(&mut block).is_ok()
                && block.id != BlockID::TrajectoryFrameSet
                && block.id != BlockID::Unknown
            {
                if block.id == block_id || block.id == BlockID::ParticleMapping {
                    self.block_read_next(&mut block, hash_mode);
                    // TODO: check stat to roll back file pos line 13898 tng_io.c
                } else {
                    file_pos += block.block_contents_size + block.header_contents_size;
                    self.input_file
                        .as_ref()
                        .expect("init input_file")
                        .seek(SeekFrom::Current(block.block_contents_size as i64))?;
                }
            }
        }
        let data = if is_particle_data {
            self.particle_data_find(block_id)
        } else {
            self.data_find(block_id)
        }
        .ok_or_else(|| TngError::NotFound("".to_string()))?;

        let (n_frames, n_particles, n_values_per_frame, data_type, current_values) = self
            .gen_data_vector_get(is_particle_data, block_id)
            .ok_or_else(|| TngError::NotFound("".to_string()))?;

        if is_particle_data && n_particles == 0 {
            return Err(TngError::NotFound("".to_string()));
        }

        // stride_length is an output of gen_data_vector_get in C; we read it from the data block
        let stride_length = data.stride_length;

        let tot_n_frames = if n_frames == 1 && n_frames < self.current_trajectory_frame_set.n_frames
        {
            1
        } else {
            end_frame_nr - start_frame_nr + 1
        };

        // Check that the reading starts at a frame with data or that the reading range is
        // long enough to reach the next frame with data of this type.
        if start_frame_nr > data.first_frame_with_data
            && start_frame_nr - data.first_frame_with_data + tot_n_frames < stride_length
        {
            return Err(TngError::Constraint("".to_string()));
        }

        if data_type == DataType::Char {
            unimplemented!("got DataType::Char unexpectedly");
        }

        // Work in f64-element space (gen_data_vector_get already converted to f64)
        let frame_size = if is_particle_data {
            (n_values_per_frame * n_particles) as usize
        } else {
            n_values_per_frame as usize
        };

        let n_frames_div_total = (tot_n_frames - 1) / stride_length + 1;
        let full_data_len = (n_frames_div_total as usize) * frame_size;
        let mut values = vec![0.0f64; full_data_len];

        if n_frames == 1 && n_frames < self.current_trajectory_frame_set.n_frames {
            values[..frame_size].copy_from_slice(&current_values[..frame_size]);
        } else {
            let current_frame_pos = start_frame_nr - self.current_trajectory_frame_set.first_frame;
            let last_frame_pos = (n_frames - 1).min(end_frame_nr - start_frame_nr);

            let n_frames_div = (current_frame_pos / stride_length) as usize;
            let n_frames_div_2 = (last_frame_pos / stride_length + 1) as usize;

            values[..n_frames_div_2 * frame_size].copy_from_slice(
                &current_values
                    [n_frames_div * frame_size..(n_frames_div + n_frames_div_2) * frame_size],
            );

            // C: current_frame_pos += n_frames - current_frame_pos  =>  = n_frames
            let mut current_frame_pos = n_frames;

            while current_frame_pos <= end_frame_nr - start_frame_nr {
                self.frame_set_read_next(hash_mode)?;

                let data = if is_particle_data {
                    self.particle_data_find(block_id)
                } else {
                    self.data_find(block_id)
                }
                .ok_or_else(|| TngError::NotFound("".to_string()))?;

                let (n_frames, _n_particles, _n_values_per_frame, _data_type, current_values) =
                    self.gen_data_vector_get(is_particle_data, block_id)
                        .ok_or_else(|| TngError::NotFound("".to_string()))?;

                let last_frame_pos = (n_frames - 1).min(end_frame_nr - current_frame_pos);

                let mut frame_pos = current_frame_pos;
                if current_frame_pos < data.first_frame_with_data
                    && end_frame_nr >= data.first_frame_with_data
                {
                    frame_pos = data.first_frame_with_data;
                }

                let n_frames_div_dst = (frame_pos / stride_length) as usize;
                let n_frames_div_2 = (last_frame_pos / stride_length + 1) as usize;

                values[n_frames_div_dst * frame_size
                    ..(n_frames_div_dst + n_frames_div_2) * frame_size]
                    .copy_from_slice(&current_values[..n_frames_div_2 * frame_size]);

                current_frame_pos += n_frames * stride_length;
            }
        }

        // *data may have been reinitialized/freed when reading frame sets. Re-find the correct data block
        if is_particle_data {
            if let Some(idx) =
                (0..self.current_trajectory_frame_set.tr_particle_data.len()).find(|&i| {
                    self.current_trajectory_frame_set.tr_particle_data[i].block_id == block_id
                })
            {
                self.current_trajectory_frame_set.tr_particle_data[idx].last_retrieved_frame =
                    end_frame_nr;
            }
        } else if let Some(idx) = (0..self.current_trajectory_frame_set.tr_data.len())
            .find(|&i| self.current_trajectory_frame_set.tr_data[i].block_id == block_id)
        {
            self.current_trajectory_frame_set.tr_data[idx].last_retrieved_frame = end_frame_nr;
        }

        Ok((
            values,
            n_particles,
            n_values_per_frame,
            stride_length,
            data_type,
        ))
    }

    fn gen_data_interval_get(
        &mut self,
        block_id: BlockID,
        is_particle_data: bool,
        start_frame_nr: i64,
        end_frame_nr: i64,
        hash_mode: bool,
    ) -> Result<(i64, i64, i64, DataType, Vec<f64>), TngError> {
        let mut block = GenBlock::new();

        let first_frame = self.current_trajectory_frame_set.first_frame;

        self.frame_set_of_frame_find(start_frame_nr)?;

        // Do not re-read the frame set.
        if (is_particle_data
            && (first_frame != self.current_trajectory_frame_set.first_frame
                || self.current_trajectory_frame_set.n_particle_data_blocks <= 0))
            || (!is_particle_data
                && (first_frame != self.current_trajectory_frame_set.first_frame
                    || self.current_trajectory_frame_set.n_data_blocks <= 0))
        {
            let mut file_pos = self.get_input_file_position();
            // Read all blocks until next frame set block
            self.block_header_read(&mut block)?;
            while file_pos < self.input_file_len
                && block.id != BlockID::TrajectoryFrameSet
                && block.id != BlockID::Unknown
            {
                self.block_read_next(&mut block, hash_mode);
                file_pos = self.get_input_file_position();
                if file_pos < self.input_file_len {
                    self.block_header_read(&mut block)?;
                }
            }
        }

        // See if there is already a data block of this ID.
        // Start checking the last read frame set
        let mut block_type_flag = false;
        let mut idx = if is_particle_data {
            let found = (0..self.current_trajectory_frame_set.n_particle_data_blocks)
                .rev()
                .find(|&i| {
                    self.current_trajectory_frame_set.tr_particle_data[i].block_id == block_id
                });
            if found.is_some() {
                block_type_flag = true;
            }
            found
        } else {
            (0..self.current_trajectory_frame_set.n_data_blocks)
                .find(|&i| self.current_trajectory_frame_set.tr_data[i].block_id == block_id)
        }
        .ok_or_else(|| {
            TngError::NotFound(format!("Could not find data block with id {block_id:?}"))
        })?;

        let n_particles = if is_particle_data {
            if block_type_flag && self.var_num_atoms {
                self.current_trajectory_frame_set.n_particles
            } else {
                self.n_particles
            }
        } else {
            0
        };

        let n_values_per_frame = if is_particle_data {
            self.current_trajectory_frame_set.tr_particle_data[idx].n_values_per_frame
        } else {
            self.current_trajectory_frame_set.tr_data[idx].n_values_per_frame
        };
        let data_type = if is_particle_data {
            self.current_trajectory_frame_set.tr_particle_data[idx].data_type
        } else {
            self.current_trajectory_frame_set.tr_data[idx].data_type
        };

        if data_type == DataType::Char {
            unimplemented!("char data not implemented for interval_get");
        }

        let n_frames = end_frame_nr - start_frame_nr + 1;
        let i_step = if is_particle_data {
            n_particles * n_values_per_frame
        } else {
            n_values_per_frame
        };
        let elem_size = data_type.get_size();
        let mut output = vec![0.0f64; (n_frames * i_step) as usize];

        let mut current_frame_pos = start_frame_nr - self.current_trajectory_frame_set.first_frame;

        for i in 0..n_frames {
            // Advance to the next frame set when the current one is exhausted.
            if current_frame_pos == self.current_trajectory_frame_set.n_frames {
                self.frame_set_read_next(hash_mode)
                    .map_err(|_| TngError::Critical("frame_set_read_next failed".into()))?;
                idx = if is_particle_data {
                    (0..self.current_trajectory_frame_set.n_particle_data_blocks)
                        .rev()
                        .find(|&i| {
                            self.current_trajectory_frame_set.tr_particle_data[i].block_id
                                == block_id
                        })
                        .ok_or_else(|| {
                            TngError::NotFound(format!(
                                "block {block_id:?} not found in next frame set"
                            ))
                        })?
                } else {
                    (0..self.current_trajectory_frame_set.n_data_blocks)
                        .find(|&i| {
                            self.current_trajectory_frame_set.tr_data[i].block_id == block_id
                        })
                        .ok_or_else(|| {
                            TngError::NotFound(format!(
                                "block {block_id:?} not found in next frame set"
                            ))
                        })?
                };
                current_frame_pos = 0;
            }

            let n_mapping_blocks = self.current_trajectory_frame_set.n_mapping_blocks;

            if is_particle_data {
                let raw = self.current_trajectory_frame_set.tr_particle_data[idx]
                    .values
                    .as_deref()
                    .unwrap_or(&[]);
                for j in 0..n_particles {
                    // Inline particle_mapping_get_real_particle to avoid borrowing self
                    // while raw is also borrowed from self.
                    let mapping = if n_mapping_blocks > 0 {
                        self.current_trajectory_frame_set
                            .mappings
                            .iter()
                            .find(|m| {
                                j >= m.num_first_particle
                                    && j < m.num_first_particle + m.n_particles
                            })
                            .map(|m| m.real_particle_numbers[(j - m.num_first_particle) as usize])
                            .unwrap_or(j)
                    } else {
                        j
                    };
                    for k in 0..n_values_per_frame {
                        let src = ((current_frame_pos * i_step + j * n_values_per_frame + k)
                            as usize)
                            * elem_size;
                        let dst = (i * i_step + mapping * n_values_per_frame + k) as usize;
                        output[dst] = interval_bytes_to_f64(&raw[src..src + elem_size], data_type);
                    }
                }
            } else {
                let raw = self.current_trajectory_frame_set.tr_data[idx]
                    .values
                    .as_deref()
                    .unwrap_or(&[]);
                for j in 0..n_values_per_frame {
                    let src = ((current_frame_pos * i_step + j) as usize) * elem_size;
                    let dst = (i * i_step + j) as usize;
                    output[dst] = interval_bytes_to_f64(&raw[src..src + elem_size], data_type);
                }
            }

            current_frame_pos += 1;
        }

        if is_particle_data {
            self.current_trajectory_frame_set.tr_particle_data[idx].last_retrieved_frame =
                end_frame_nr;
        } else {
            self.current_trajectory_frame_set.tr_data[idx].last_retrieved_frame = end_frame_nr;
        }

        Ok((n_frames, n_particles, n_values_per_frame, data_type, output))
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
                self.block_read_next(&mut block, USE_HASH);
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

                    let size_i64 = i64::try_from(size).expect("size to i64");
                    let src_base = ((i * n_particles + j) * n_values_per_frame * size_i64) as usize;
                    let dst_base =
                        ((i * n_particles + mapping) * n_values_per_frame * size_i64) as usize;

                    values[dst_base..dst_base + byte_per_particle]
                        .copy_from_slice(&unwrapped_values[src_base..src_base + byte_per_particle]);
                }
            }
        }

        let float_values: Vec<f64> = match data_type {
            DataType::Char => unimplemented!("haven't implemented values to strings"),
            DataType::Int => values
                .chunks_exact(8)
                .map(|chunk| {
                    let arr = <[u8; 8]>::try_from(chunk).expect("Chunk should be 8 bytes");
                    i64::from_ne_bytes(arr) as f64
                })
                .collect(),
            DataType::Float => values
                .chunks_exact(4)
                .map(|chunk| {
                    let arr = <[u8; 4]>::try_from(chunk).expect("Chunk should be 4 bytes");
                    f32::from_ne_bytes(arr) as f64
                })
                .collect(),
            DataType::Double => values
                .chunks_exact(8)
                .map(|chunk| {
                    let arr = <[u8; 8]>::try_from(chunk).expect("Chunk should be 8 bytes");
                    f64::from_ne_bytes(arr)
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

    /// C API: `tng_frame_set_read_next`.
    ///
    /// Read one (the next) frame set, including particle mapping and related data blocks
    /// from the input_file of [`Self`]
    pub fn frame_set_read_next(&mut self, hash_mode: bool) -> Result<(), TngError> {
        self.input_file_init();

        let mut file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
        if file_pos < 0 && self.current_trajectory_frame_set_input_file_pos <= 0 {
            file_pos = self.first_trajectory_frame_set_input_file_pos;
        }

        if file_pos > 0 {
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("i64 to u64"),
                ))?;
        } else {
            return Err(TngError::Constraint(format!(
                "file_pos was negative. Got {file_pos}"
            )));
        }
        self.frame_set_read(hash_mode)
    }

    /// Read one frame set, including all particle mapping blocks and data blocks, starting from
    /// the current file position
    fn frame_set_read(&mut self, hash_mode: bool) -> Result<(), TngError> {
        self.input_file_init();
        let mut file_pos = self.get_input_file_position();
        let mut block = GenBlock::new();

        // Read block headers first to see what block is found
        self.block_header_read(&mut block)?;
        if block.id != BlockID::TrajectoryFrameSet || block.id == BlockID::Unknown {
            return Err(TngError::Critical(format!(
                "Cannot read block header at pos {file_pos}"
            )));
        }

        self.current_trajectory_frame_set_input_file_pos =
            i64::try_from(file_pos).expect("u64 to i64");

        // TODO: make this fallible?
        self.block_read_next(&mut block, hash_mode);
        if block.id != BlockID::Unknown {
            self.n_trajectory_frame_sets += 1;
            file_pos = self.get_input_file_position();

            // Read all blocks until next frame set block
            self.block_header_read(&mut block)?;
            loop {
                if file_pos >= self.input_file_len {
                    break;
                }
                match block.id {
                    BlockID::Unknown | BlockID::TrajectoryFrameSet => break,
                    _ => {}
                }
                self.block_read_next(&mut block, hash_mode);
                file_pos = self.get_input_file_position();
                if file_pos < self.input_file_len {
                    self.block_header_read(&mut block)?;
                }
            }

            if block.id == BlockID::TrajectoryFrameSet {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(file_pos))?;
            }
        }
        Ok(())
    }

    /// C API: `tng_frame_set_write`.
    ///
    /// Write one frame set, including mapping and related data blocks to [`self.output_file`]
    /// of [`Self`]
    pub fn frame_set_write(&mut self, hash_mode: bool) -> Result<(), TngError> {
        let frame_set = &self.current_trajectory_frame_set;
        if frame_set.n_written_frames == frame_set.n_frames {
            return Ok(());
        }

        self.current_trajectory_frame_set_output_file_pos =
            i64::try_from(self.get_output_file_position()).expect("i64 from u64");
        self.last_trajectory_frame_set_output_file_pos =
            self.current_trajectory_frame_set_output_file_pos;

        if self.current_trajectory_frame_set_output_file_pos <= 0 {
            return Err(TngError::Constraint(
                "output file position is invalid; cannot write frame set".to_string(),
            ));
        }

        if self.first_trajectory_frame_set_output_file_pos == -1 {
            self.first_trajectory_frame_set_output_file_pos =
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
                    // TODO: error handling
                    self.trajectory_mapping_block_write(&mut block, i, hash_mode)
                        .expect("handle errors");
                    for j in 0..self.current_trajectory_frame_set.n_particle_data_blocks {
                        block.id = self.current_trajectory_frame_set.tr_particle_data[j].block_id;
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
        let mut file_pos = self.first_trajectory_frame_set_input_file_pos;

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

        self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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
                u64::try_from(self.first_trajectory_frame_set_input_file_pos).expect("i64 to u64"),
            ))
            .expect("no error handling");
        self.current_trajectory_frame_set_input_file_pos = orig_frame_set_file_pos;

        count
    }

    /// C API: `tng_frame_set_nr_find`.
    ///
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
            self.first_trajectory_frame_set_input_file_pos
        } else {
            // Start from the end
            curr_nr = n_frame_sets - 1;
            self.last_trajectory_frame_set_input_file_pos
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

        self.block_read_next(&mut block, SKIP_HASH);

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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
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

                self.block_read_next(&mut block, SKIP_HASH);
                if curr_nr == nr {
                    return Ok(());
                }
            }
        }

        Err(())
    }

    /// C API: `tng_frame_set_read_current_only_data_from_block_id`.
    ///
    /// Read data from the current frame set from the `input_file`. Only read
    /// particle mapping and data blocks matching the specified [`BlockID`]
    pub fn frame_set_read_current_only_data_from_block_id(
        &mut self,
        hash_mode: bool,
        match_block_id: BlockID,
    ) -> Result<(), TngError> {
        let mut found_flag = false;
        self.input_file_init();

        let mut file_pos = self.current_trajectory_frame_set_input_file_pos;

        if file_pos < 0 {
            // No current frame set. This means that the first frame set must be read
            found_flag = true;
            file_pos = self.first_trajectory_frame_set_input_file_pos;
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
            return Err(TngError::Constraint(format!(
                "`file_pos` was negative. Got {file_pos}"
            )));
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
            self.block_read_next(&mut block, hash_mode);
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
                self.block_read_next(&mut block, hash_mode);
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

        if found_flag {
            Ok(())
        } else {
            Err(TngError::NotFound("".to_string()))
        }
    }

    /// C API: `tng_frame_set_read_next_only_data_from_block_id`.
    ///
    /// Read one (the next) frame set, including particle mapping and data blocks with
    /// a specific block id from `input_file` of [`Self`]
    pub fn frame_set_read_next_only_data_from_block_id(
        &mut self,
        hash_mode: bool,
        match_block_id: BlockID,
    ) -> Result<(), TngError> {
        self.input_file_init();

        let mut file_pos = self.current_trajectory_frame_set.next_frame_set_file_pos;
        if file_pos < 0 && self.current_trajectory_frame_set_input_file_pos <= 0 {
            file_pos = self.first_trajectory_frame_set_input_file_pos;
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
            return Err(TngError::Constraint(format!(
                "`file_pos` was negative. Got {file_pos}"
            )));
        }

        let mut block = GenBlock::new();

        // Read block headers first to see what block is found
        self.block_header_read(&mut block);
        if block.id != BlockID::TrajectoryFrameSet {
            panic!("Cannot read block header at pos {file_pos}");
        }

        self.current_trajectory_frame_set_input_file_pos = file_pos;

        self.block_read_next(&mut block, hash_mode);

        self.frame_set_read_current_only_data_from_block_id(hash_mode, match_block_id)
    }

    /// C API: `tng_data_block_name_get`.
    ///
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
                let result =
                    self.frame_set_read_current_only_data_from_block_id(USE_HASH, match_block_id);
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

    /// C API: `tng_data_block_dependency_get`.
    ///
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
                let result =
                    self.frame_set_read_current_only_data_from_block_id(USE_HASH, match_block_id);
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

    /// C API: `tng_data_block_num_values_per_frame_get`.
    ///
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
                let stat =
                    self.frame_set_read_current_only_data_from_block_id(USE_HASH, match_block_id);
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
            self.first_trajectory_frame_set_input_file_pos
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

    /// C API: `tng_num_frames_get`.
    pub fn num_frames_get(&mut self) -> Option<i64> {
        let file_pos = self.get_input_file_position();
        let last_file_pos = self.last_trajectory_frame_set_input_file_pos;

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
    /// [`Self::current_trajectory_frame_set`] will contain the found trajectory if successful.
    pub fn frame_set_of_frame_find(&mut self, frame: i64) -> Result<(), TngError> {
        let mut block = GenBlock::new();
        let mut file_pos = 0;
        if self.current_trajectory_frame_set_input_file_pos < 0 {
            file_pos = self.first_trajectory_frame_set_input_file_pos;
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    u64::try_from(file_pos).expect("u64 from i64"),
                ))
                .expect("no error handling");
            self.current_trajectory_frame_set_input_file_pos = file_pos;

            // Read block headers first to see what block is found
            self.block_header_read(&mut block)?;
            if block.id != BlockID::TrajectoryFrameSet {
                panic!("Cannot read block header at pos {file_pos}");
            }

            self.block_read_next(&mut block, SKIP_HASH);
        }

        let frame_set = &self.current_trajectory_frame_set;
        let first_frame = max(frame_set.first_frame, 0);
        let last_frame = first_frame + frame_set.n_frames - 1;

        // Is this the right frame set?
        if first_frame <= frame && frame <= last_frame {
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

        if frame >= n_frames {
            return Err(TngError::NotFound(format!(
                "Asked for a frame ({frame}) beyond what is currently available ({n_frames})"
            )));
        }

        if first_frame - frame >= frame
            || frame - last_frame > self.n_trajectory_frame_sets * n_frames_per_frame_set
        {
            // Start from the beginning
            if first_frame - frame >= frame {
                file_pos = self.first_trajectory_frame_set_input_file_pos;
                if file_pos <= 0 {
                    return Err(TngError::NotFound(format!(
                        "file_pos is below or equal to 0 (file_pos: {file_pos})"
                    )));
                }
            } else if frame - first_frame > (n_frames - 1) - frame {
                file_pos = self.last_trajectory_frame_set_input_file_pos;
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }
        }

        let mut first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
        let mut last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;

        if frame >= first_frame && frame <= last_frame {
            return Ok(());
        }

        file_pos = self.current_trajectory_frame_set_input_file_pos;

        // Take long steps forward until a long step forward would be too long or
        // the right frame is found
        while file_pos > 0 && first_frame + long_stride_length * n_frames_per_frame_set <= frame {
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if frame >= first_frame && frame <= last_frame {
                return Ok(());
            }
        }

        // Take medium steps forward until a medium step forward would be too long or
        // the right frame is found
        while file_pos > 0 && first_frame + medium_stride_length * n_frames_per_frame_set <= frame {
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if frame >= first_frame && frame <= last_frame {
                return Ok(());
            }
        }

        // Take one step forward until the right frame is found
        while file_pos > 0 && first_frame < frame && last_frame < frame {
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if frame >= first_frame && frame <= last_frame {
                return Ok(());
            }
        }

        // Take long steps backward until a long step backward would be too long or
        // the right frame is found
        while file_pos > 0 && first_frame - long_stride_length * n_frames_per_frame_set >= frame {
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if frame >= first_frame && frame <= last_frame {
                return Ok(());
            }
        }

        // Take medium steps backward until a medium step backward would be too long or
        // the right frame is found
        while file_pos > 0 && first_frame - medium_stride_length * n_frames_per_frame_set >= frame {
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if frame >= first_frame && frame <= last_frame {
                return Ok(());
            }
        }

        // Take one step backward until the right frame is found
        while file_pos > 0 && first_frame > frame && last_frame > frame {
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if frame >= first_frame && frame <= last_frame {
                return Ok(());
            }
        }

        // If for some reason the current frame set is not yet found
        // take one step forward until the right frame set is found
        while file_pos > 0 && first_frame < frame && last_frame < frame {
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
                self.block_header_read(&mut block)?;
                if block.id != BlockID::TrajectoryFrameSet {
                    panic!("Cannot read block header at pos {file_pos}");
                }

                self.block_read_next(&mut block, SKIP_HASH);
            }

            first_frame = max(self.current_trajectory_frame_set.first_frame, 0);
            last_frame = first_frame + self.current_trajectory_frame_set.n_frames - 1;
            if frame >= first_frame && frame <= last_frame {
                return Ok(());
            }
        }

        Err(TngError::NotFound(format!("Could not find frame: {frame}")))
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
            return Err(TngError::Critical(
                "Cannot read general info header.".to_string(),
            ));
        };

        let output_file = self
            .output_file
            .as_mut()
            .expect("just initialized output file");

        // TODO: hash mode
        let _contents_start_pos = output_file.stream_position().expect("no error handling");
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

        let mut pos = self.first_trajectory_frame_set_output_file_pos;
        utils::write_u64(
            output_file,
            u64::try_from(pos).expect("u64 from i64"),
            self.endianness64,
            self.input_swap64,
        );

        pos = self.last_trajectory_frame_set_output_file_pos;
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

        self.block_read_next(&mut block, SKIP_HASH);
    }

    /// C API: `tng_data_get_stride_length`.
    ///
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
                let mut stat =
                    self.frame_set_read_current_only_data_from_block_id(USE_HASH, match_block_id);

                // If no specific frame was required read until this data block is found
                if new_frame < 0 {
                    let mut file_pos = self.get_input_file_position();

                    while stat.is_err() && file_pos < self.input_file_len {
                        stat = self
                            .frame_set_read_next_only_data_from_block_id(USE_HASH, match_block_id);
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

    /// High-level function for opening and initializing a TNG trajectory
    pub fn util_trajectory_open(&mut self, filename: &Path, mode: char) -> Result<(), TngError> {
        match mode {
            'r' | 'w' | 'a' => {}
            _ => {
                return Err(TngError::Constraint(format!(
                    "mode must one of 'r', 'w', or 'a'. Got {mode}"
                )));
            }
        };

        // TODO: does this even make sense? we can't call this method on traj without already having a traj
        *self = Trajectory::new();

        if mode == 'w' {
            self.output_file_set(filename);
        }
        self.input_file_set(filename);

        // Read the file headers
        self.file_headers_read(USE_HASH);

        let n = self.num_frame_sets_get();
        self.n_trajectory_frame_sets = n;

        if mode == 'a' {
            // If a file was already open, drop (close) it.
            self.output_file.take();
            self.output_file = Some(
                self.input_file
                    .as_mut()
                    .expect("we just set the input file")
                    .try_clone()?,
            );
            self.input_file
                .as_mut()
                .expect("init input_file")
                .seek(SeekFrom::Start(
                    self.last_trajectory_frame_set_input_file_pos as u64,
                ))?;

            self.frame_set_read(USE_HASH)?;

            self.output_file = None;

            self.first_trajectory_frame_set_output_file_pos =
                self.first_trajectory_frame_set_input_file_pos;
            self.last_trajectory_frame_set_output_file_pos =
                self.last_trajectory_frame_set_input_file_pos;
            self.current_trajectory_frame_set_output_file_pos =
                self.current_trajectory_frame_set_input_file_pos;

            // If a file was already open, drop (close) it.
            self.input_file.take();
            self.input_file_path.take();
            self.output_append_file_set(filename)?;
        }

        Ok(())
    }

    /// C API: `tng_molecule_add`.
    ///
    /// Add a molecule to the trajectory.
    /// Returns the index of the new molecule in `self.molecules`.
    pub fn add_molecule(&mut self, name: &str) -> usize {
        let id = self.molecules.last().map(|m| m.id + 1).unwrap_or(1);
        self.molecule_w_id_add(name, id)
    }

    /// C API: `tng_molecule_w_id_add`
    ///
    /// Add a molecule with a specific ID to the trajectory
    fn molecule_w_id_add(&mut self, name: &str, id: i64) -> usize {
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

    /// C API: `tng_molecule_chain_add`.
    ///
    /// Add a chain to the molecule at `molecule_idx`.
    /// Returns the index of the new chain in `self.molecules[molecule_idx].chains`.
    pub fn add_chain(&mut self, molecule_idx: usize, name: &str) -> usize {
        let id = self.molecules[molecule_idx]
            .chains
            .last()
            .map(|c| c.id + 1)
            .unwrap_or(1);
        self.chain_w_id_add(molecule_idx, name, id)
    }

    /// C API: `tng_molecule_chain_w_id_add`
    ///
    /// Add a chain with a specific id to a molecule
    fn molecule_chain_w_id_add(&self, molecule: &mut Molecule, name: &str, id: u64) {
        let mut new_chain = Chain::default();
        new_chain.name = String::new();

        new_chain.set_name(name);
        new_chain.parent_molecule_idx = molecule.id as usize;
        new_chain.n_residues = 0;
        molecule.n_chains += 1;

        new_chain.id = id;

        molecule.chains.push(new_chain);
    }

    /// C API: `tng_chain_w_id_add`
    fn chain_w_id_add(&mut self, molecule_idx: usize, name: &str, id: u64) -> usize {
        let mut chain = Chain::new();
        chain.set_name(name);
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

    /// C API: `tng_chain_residue_add`.
    ///
    /// Add a residue to the molecule at `molecule_idx`, associated with chain `chain_idx`.
    /// Returns the index of the new residue in `self.molecules[molecule_idx].residues`.
    pub fn chain_residue_add(
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
        self.chain_residue_w_id_add(molecule_idx, chain_idx, name, id)
    }

    /// C API: `tng_chain_residue_w_id_add`
    ///
    /// Add a residue with a specific ID to a chain.
    fn chain_residue_w_id_add(
        &mut self,
        molecule_idx: usize,
        chain_idx: usize,
        name: &str,
        id: u64,
    ) -> usize {
        let mol = &mut self.molecules[molecule_idx];

        let insert_pos = mol.chains[chain_idx].residues_indices.1;

        let mut residue = Residue::new();
        residue.parent_molecule_idx = molecule_idx;
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
            if let Some(ri) = atom.residue_index
                && ri >= insert_pos
            {
                atom.residue_index = Some(ri + 1);
            }
        }

        mol.n_residues += 1;
        insert_pos
    }

    /// C API: `tng_residue_atom_add`.
    ///
    /// Add an atom to the molecule at `molecule_idx`, associated with residue `residue_idx`.
    /// Returns the index of the new atom in `self.molecules[molecule_idx].atoms`.
    pub fn residue_atom_add(
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
        self.residue_atom_w_id_add(molecule_idx, residue_idx, name, atom_type, id)
    }

    /// C API: `tng_residue_atom_w_id_add`
    ///
    /// Add an atom with a specific ID to a residue
    fn residue_atom_w_id_add(
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
        atom.parent_molecule_idx = molecule_idx;
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

    /// C API: `tng_molecule_bond_add`.
    ///
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

    /// C API: `tng_data_block_add`.
    pub fn add_data_block(
        &mut self,
        id: BlockID,
        block_name: &str,
        data_type: DataType,
        block_type_flag: &BlockType,
        n_frames: i64,
        n_values_per_frame: i64,
        stride_length: i64,
        codec_id: Compression,
        new_data: Option<&[u8]>,
    ) -> Result<(), TngError> {
        if n_values_per_frame <= 0 {
            return Err(TngError::Constraint(format!(
                "`n_values_per_frame` must be a positive integer. Got {n_values_per_frame}"
            )));
        }

        self.gen_data_block_add(
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

    fn gen_data_block_add(
        &mut self,
        id: BlockID,
        is_particle_data: bool,
        block_name: &str,
        data_type: DataType,
        block_type_flag: &BlockType,
        n_frames: i64,
        n_values_per_frame: i64,
        stride_length: i64,
        num_first_particle: u64,
        n_particles: i64,
        codec_id: Compression,
        new_data: Option<&[u8]>,
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

        let is_new_block = found_block.is_none();
        let mut data = if let Some(existing) = found_block {
            existing
        } else {
            // If the block does not exist, create it
            if is_particle_data {
                self.particle_data_block_create(&block_type_flag);
            } else {
                self.data_block_create(&block_type_flag);
            }

            let mut data = Data {
                block_id: id,
                ..Default::default()
            };
            let length = block_name.floor_char_boundary(MAX_STR_LEN - 1);
            data.block_name = block_name[..length].to_string();
            data.values = None;
            data.strings = None;
            data.last_retrieved_frame = -1;
            data
        };
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
        if block_type_flag == &BlockType::Trajectory
            && (n_frames > 1 || frame_set.n_frames == n_frames || stride_length > 1)
        {
            data.dependency |= FRAME_DEPENDENT;
        }
        data.codec_id = codec_id;
        data.compression_multiplier = 1.0;
        // FIXME(from C code): this can cause problems
        data.first_frame_with_data = frame_set.first_frame;

        let mut tot_n_particles = 0;
        if is_particle_data {
            tot_n_particles = if block_type_flag == &BlockType::Trajectory && self.var_num_atoms {
                frame_set.n_particles
            } else {
                self.n_particles
            };
        }

        // If data values are supplied add that data to the data block
        if let Some(new_data) = new_data {
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

        // Write the modified data back to the appropriate store.
        // The local `data` was cloned from the store (or created as default),
        // so we must put it back for changes to take effect.
        if is_new_block {
            // We just pushed a Data::default() — replace it with the fully initialized one
            if is_particle_data {
                if block_type_flag == &BlockType::Trajectory {
                    *self
                        .current_trajectory_frame_set
                        .tr_particle_data
                        .last_mut()
                        .expect("just created") = data;
                } else {
                    *self.non_tr_particle_data.last_mut().expect("just created") = data;
                }
            } else if block_type_flag == &BlockType::Trajectory {
                *self
                    .current_trajectory_frame_set
                    .tr_data
                    .last_mut()
                    .expect("just created") = data;
            } else {
                *self.non_tr_data.last_mut().expect("just created") = data;
            }
        } else {
            // Update the existing block in-place by finding it by ID
            let target = if is_particle_data {
                if block_type_flag == &BlockType::Trajectory {
                    self.current_trajectory_frame_set
                        .tr_particle_data
                        .iter_mut()
                        .find(|d| d.block_id == id)
                } else {
                    self.non_tr_particle_data
                        .iter_mut()
                        .find(|d| d.block_id == id)
                }
            } else if block_type_flag == &BlockType::Trajectory {
                self.current_trajectory_frame_set
                    .tr_data
                    .iter_mut()
                    .find(|d| d.block_id == id)
                    .or_else(|| self.non_tr_data.iter_mut().find(|d| d.block_id == id))
            } else {
                self.non_tr_data.iter_mut().find(|d| d.block_id == id)
            };
            if let Some(existing) = target {
                *existing = data;
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

    /// C API: `tng_particle_data_block_add`.
    pub(crate) fn particle_data_block_add(
        &mut self,
        id: BlockID,
        block_name: &str,
        data_type: DataType,
        block_type_flag: &BlockType,
        n_frames: i64,
        n_values_per_frame: i64,
        stride_length: i64,
        num_first_particle: u64,
        n_particles: i64,
        codec_id: Compression,
        new_data: Option<&[u8]>,
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

        self.gen_data_block_add(
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

    /// C API: `tng_num_molecules_get`.
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

    /// C API: `tng_num_frames_per_frame_set_get`.
    pub(crate) fn num_frames_per_frame_set_get(&self) -> i64 {
        self.frame_set_n_frames
    }

    /// C API: `tng_frame_set_new`.
    pub(crate) fn frame_set_new(
        &mut self,
        first_frame: i64,
        n_frames: i64,
    ) -> Result<(), TngError> {
        if first_frame < 0 {
            return Err(TngError::Constraint(format!(
                "`first_frame` must be >= 0. Got {first_frame}"
            )));
        }
        if n_frames < 0 {
            return Err(TngError::Constraint(format!(
                "`n_frames` must be >= 0. Got {n_frames}"
            )));
        }

        self.output_file_init();

        let curr_file_pos = self.get_output_file_position();

        if curr_file_pos <= 10 {
            self.file_headers_write(USE_HASH)?;
        }

        // Set pointer to previous frame set to the one that was loaded before.
        // FIXME(from c code): This is a bit risky. If they are not added in order it will be wrong
        if self.n_trajectory_frame_sets > 0 {
            self.current_trajectory_frame_set.prev_frame_set_file_pos =
                self.last_trajectory_frame_set_output_file_pos;
        }

        self.current_trajectory_frame_set.next_frame_set_file_pos = -1;

        self.current_trajectory_frame_set_output_file_pos = self.get_output_file_position() as i64;

        self.n_trajectory_frame_sets += 1;

        // Set the medium range pointers
        if self.n_trajectory_frame_sets == self.medium_stride_length + 1 {
            self.current_trajectory_frame_set
                .medium_stride_prev_frame_set_file_pos =
                self.first_trajectory_frame_set_output_file_pos;
        } else if self.n_trajectory_frame_sets > self.medium_stride_length + 1 {
            // FIXME(from c code): Currently only working if the previous frame set has its
            // medium stride pointer already set.
            let medium_prev = self
                .current_trajectory_frame_set
                .medium_stride_prev_frame_set_file_pos;
            if medium_prev != -1 && medium_prev != 0 {
                let mut block = GenBlock::new();

                // Temporarily use output file as input to read the block header
                let temp_input = self.input_file.take();
                self.input_file = self.output_file.take();

                let curr_file_pos = self.get_input_file_position();
                self.input_file
                    .as_mut()
                    .expect("init input_file")
                    .seek(SeekFrom::Start(medium_prev as u64))?;

                if let Err(e) = self.block_header_read(&mut block) {
                    eprintln!("Cannot read frame set header. {}:{}", file!(), line!());
                    self.output_file = self.input_file.take();
                    self.input_file = temp_input;
                    return Err(e);
                }

                // Read the next frame set from the previous frame set and one
                // medium stride step back.
                // Skip to medium_stride_next field: block_contents_size - 6*i64 - 2*f64
                let skip = block.block_contents_size as i64
                    - (6 * std::mem::size_of::<i64>() as i64
                        + 2 * std::mem::size_of::<f64>() as i64);
                self.input_file
                    .as_mut()
                    .expect("init input_file")
                    .seek(SeekFrom::Current(skip))?;

                self.current_trajectory_frame_set
                    .medium_stride_prev_frame_set_file_pos = utils::read_i64(
                    self.input_file.as_mut().expect("init input_file"),
                    self.endianness64,
                    self.input_swap64,
                );

                // Set the long range pointers
                if self.n_trajectory_frame_sets == self.long_stride_length + 1 {
                    self.current_trajectory_frame_set
                        .long_stride_prev_frame_set_file_pos =
                        self.first_trajectory_frame_set_output_file_pos;
                } else if self.n_trajectory_frame_sets > self.medium_stride_length + 1 {
                    let long_prev = self
                        .current_trajectory_frame_set
                        .long_stride_prev_frame_set_file_pos;
                    if long_prev != -1 && long_prev != 0 {
                        let mut block = GenBlock::new();

                        self.input_file
                            .as_mut()
                            .expect("init input_file")
                            .seek(SeekFrom::Start(long_prev as u64))?;

                        if let Err(e) = self.block_header_read(&mut block) {
                            eprintln!("Cannot read frame set header. {}:{}", file!(), line!());
                            self.output_file = self.input_file.take();
                            self.input_file = temp_input;
                            return Err(e);
                        }

                        // Skip to long_stride_next field
                        let skip = block.block_contents_size as i64
                            - (6 * std::mem::size_of::<i64>() as i64
                                + 2 * std::mem::size_of::<f64>() as i64);
                        self.input_file
                            .as_mut()
                            .expect("init input_file")
                            .seek(SeekFrom::Current(skip))?;

                        self.current_trajectory_frame_set
                            .long_stride_prev_frame_set_file_pos = utils::read_i64(
                            self.input_file.as_mut().expect("init input_file"),
                            self.endianness64,
                            self.input_swap64,
                        );
                    }
                }

                // Restore input/output files and seek back to original position
                self.output_file = self.input_file.take();
                self.input_file = temp_input;
                self.output_file
                    .as_mut()
                    .expect("init output_file")
                    .seek(SeekFrom::Start(curr_file_pos))?;
            }
        }

        let frame_set = &mut self.current_trajectory_frame_set;
        frame_set.first_frame = first_frame;
        frame_set.n_frames = n_frames;
        frame_set.n_written_frames = 0;
        frame_set.n_unwritten_frames = 0;
        frame_set.first_frame_time = -1.0;

        if self.first_trajectory_frame_set_output_file_pos == -1
            || self.first_trajectory_frame_set_output_file_pos == 0
        {
            self.first_trajectory_frame_set_output_file_pos =
                self.current_trajectory_frame_set_output_file_pos;
        }

        if self.last_trajectory_frame_set_output_file_pos == -1
            || self.last_trajectory_frame_set_output_file_pos == 0
            || self.last_trajectory_frame_set_output_file_pos
                < self.current_trajectory_frame_set_output_file_pos
        {
            self.last_trajectory_frame_set_output_file_pos =
                self.current_trajectory_frame_set_output_file_pos;
        }

        Ok(())
    }

    /// C API: `tng_frame_set_with_time_new`.
    pub(crate) fn frame_set_with_time_new(
        &mut self,
        first_frame: i64,
        n_frames: i64,
        first_frame_time: f64,
    ) -> Result<(), TngError> {
        if first_frame_time < 0.0 {
            return Err(TngError::Constraint(
                "`first_frame_time` must be >= 0".to_string(),
            ));
        }

        self.frame_set_new(first_frame, n_frames)?;
        self.current_trajectory_frame_set.first_frame_time = first_frame_time;

        Ok(())
    }

    /// C API: `tng_particle_mapping_add`.
    pub(crate) fn particle_mapping_add(
        &mut self,
        num_first_particle: i64,
        n_particles: i64,
        mapping_table: &[i64],
    ) -> Result<(), TngError> {
        let frame_set = &mut self.current_trajectory_frame_set;

        // Sanity check of the particles ranges. Split into multiple if
        // statements for improved readability
        for i in 0..frame_set.n_mapping_blocks {
            let mapping = &frame_set.mappings[i as usize];
            if num_first_particle >= mapping.num_first_particle
                && num_first_particle < mapping.num_first_particle + mapping.n_particles
            {
                return Err(TngError::Constraint(
                    "Particle mapping overlap.".to_string(),
                ));
            }
            if num_first_particle + n_particles >= mapping.num_first_particle
                && num_first_particle + n_particles
                    < mapping.num_first_particle + mapping.n_particles
            {
                return Err(TngError::Constraint("Particle mapping overlap".to_string()));
            }
            if mapping.num_first_particle >= num_first_particle
                && mapping.num_first_particle < num_first_particle + n_particles
            {
                return Err(TngError::Constraint("Particle mapping overlap".to_string()));
            }
            if mapping.num_first_particle + mapping.n_particles > num_first_particle
                && mapping.num_first_particle + mapping.n_particles
                    < num_first_particle + n_particles
            {
                return Err(TngError::Constraint("Particle mapping overlap".to_string()));
            }
        }

        frame_set.n_mapping_blocks += 1;
        frame_set.mappings.push(ParticleMapping::new());

        frame_set.mappings[(frame_set.n_mapping_blocks - 1) as usize].num_first_particle =
            num_first_particle;
        frame_set.mappings[(frame_set.n_mapping_blocks - 1) as usize].n_particles = n_particles;
        for item in mapping_table.iter().take(n_particles as usize) {
            frame_set.mappings[(frame_set.n_mapping_blocks - 1) as usize]
                .real_particle_numbers
                .push(*item);
        }

        Ok(())
    }

    /// C API: `tng_frame_set_particle_mapping_free`.
    pub(crate) fn frame_set_particle_mapping_free(&mut self) {
        let frame_set = &mut self.current_trajectory_frame_set;

        if frame_set.n_mapping_blocks > 0 && !frame_set.mappings.is_empty() {
            for i in 0..frame_set.n_mapping_blocks as usize {
                let mapping = &mut frame_set.mappings[i];
                mapping.real_particle_numbers = vec![];
            }
            frame_set.mappings = vec![];
            frame_set.n_mapping_blocks = 0;
        }
    }

    fn validate_get_name_len<'a>(
        name: &'a str,
        field_name: &str,
        max_len: usize,
    ) -> Result<&'a str, TngError> {
        if max_len == 0 {
            return Err(TngError::Constraint(format!(
                "{field_name} get requires `max_len` > 0"
            )));
        }

        if name.len() > max_len - 1 {
            return Err(TngError::Constraint(format!(
                "{field_name} was longer than `max_len` (`max_len` = {max_len})"
            )));
        }

        Ok(name)
    }
    /// C API: `tng_last_program_name_get`.
    ///
    /// Get the name of the program used when last modifying the trajectory.
    pub fn last_program_name_get(&self, max_len: usize) -> Result<&str, TngError> {
        Self::validate_get_name_len(&self.last_program_name, "first user name", max_len)
    }

    /// C API: `tng_last_program_name_set`.
    ///
    /// Set the name of the program used when last modifying the trajectory.
    pub fn last_program_name_set(&mut self, new_name: &str) {
        // We use `floor_char_boundary` as Rust strings has to be valid UTF-8. This way we never split a charachter
        // and never panic on non-UTF-8 names
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.last_program_name = new_name[..length].to_string();
    }

    /// C API: `tng_first_user_name_get`.
    ///
    /// Get the name of the user who created the trajectory
    pub fn first_user_name_get(&self, max_len: usize) -> Result<&str, TngError> {
        Self::validate_get_name_len(&self.first_user_name, "first user name", max_len)
    }

    /// C API: `tng_first_user_name_set`.
    ///
    /// Set the name of the user who created the trajectory
    pub fn first_user_name_set(&mut self, new_name: &str) {
        // We use `floor_char_boundary` as Rust strings has to be valid UTF-8. This way we never split a charachter
        // and never panic on non-UTF-8 names
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.first_user_name = new_name[..length].to_string();
    }

    /// C API: `tng_last_user_name_get`.
    ///
    /// Get the name of the user who last modified the trajectory
    pub fn last_user_name_get(&self, max_len: usize) -> Result<&str, TngError> {
        Self::validate_get_name_len(&self.last_user_name, "last user name", max_len)
    }

    /// C API: `tng_last_user_name_set`.
    ///
    /// Set the name of the user who last modified the trajectory
    pub fn last_user_name_set(&mut self, new_name: &str) {
        // We use `floor_char_boundary` as Rust strings has to be valid UTF-8. This way we never split a charachter
        // and never panic on non-UTF-8 names
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.last_user_name = new_name[..length].to_string();
    }

    /// C API: `tng_first_program_name_get`.
    pub fn first_program_name_get(&self, max_len: usize) -> Result<&str, TngError> {
        Self::validate_get_name_len(&self.first_program_name, "first program name", max_len)
    }

    /// C API: `tng_first_computer_name_get`.
    ///
    /// Get the name of the computer used when creating the trajectory
    pub fn first_computer_name_get(&self, max_len: usize) -> Result<&str, TngError> {
        Self::validate_get_name_len(&self.first_computer_name, "first computer name", max_len)
    }

    /// C API: `tng_last_computer_name_get`.
    ///
    /// Get the name of the computer used when last modifying the trajectory.
    pub fn last_computer_name_get(&self, max_len: usize) -> Result<&str, TngError> {
        Self::validate_get_name_len(&self.last_computer_name, "first user name", max_len)
    }

    /// C API: `tng_last_computer_name_set`.
    ///
    /// Set the name of the computer used when last modifying the trajectory.
    pub fn last_computer_name_set(&mut self, new_name: &str) {
        // We use `floor_char_boundary` as Rust strings has to be valid UTF-8. This way we never split a charachter
        // and never panic on non-UTF-8 names
        let length = new_name.floor_char_boundary(MAX_STR_LEN - 1);
        self.last_computer_name = new_name[..length].to_string();
    }

    /// C API: `tng_forcefield_name_get`.
    pub fn forcefield_name_get(&self, max_len: usize) -> Result<&str, TngError> {
        Self::validate_get_name_len(&self.forcefield_name, "forcefield name", max_len)
    }

    /// C API: `tng_medium_stride_length_get`.
    pub fn medium_stride_length_get(&self) -> i64 {
        self.medium_stride_length
    }

    /// C API: `tng_long_stride_length_get`.
    pub fn long_stride_length_get(&self) -> i64 {
        self.long_stride_length
    }

    /// C API: `tng_compression_precision_get`.
    pub fn compression_precision_get(&self) -> f64 {
        self.compression_precision
    }

    /// C API: `tng_distance_unit_exponential_get`.
    pub fn distance_unit_exponential_get(&self) -> i64 {
        self.distance_unit_exponential
    }

    /// C API: `tng_num_molecule_types_get`.
    pub fn num_molecule_types_get(&self) -> i64 {
        self.n_molecules
    }

    /// C API: `tng_num_molecules_get`.
    pub fn num_molecules_get(&self) -> i64 {
        let cnt_list = self.molecule_cnt_list_get();

        cnt_list.iter().take(self.n_molecules as usize).sum()
    }

    /// C API: `tng_num_particles_variable_get`.
    pub(crate) fn num_particles_variable_get(&self) -> bool {
        self.var_num_atoms
    }

    /// C API: `tng_molecule_of_index_get`.
    ///
    /// # Errors
    ///
    /// Returns [`TngError::NotFound`] if `index` is bigger than the amount of molecules
    pub(crate) fn molecule_of_index_get(&self, index: i64) -> Result<&Molecule, TngError> {
        if index >= self.n_molecules || index < 0 {
            return Err(TngError::NotFound(format!(
                "A molecule with index {index} was not found."
            )));
        }
        Ok(&self.molecules[index as usize])
    }

    /// C API: `tng_molecule_find`.
    /// # Errors
    ///
    /// Returns [`TngError::NotFound`] if the molecule cannot be found
    pub fn molecule_find(
        &self,
        name: Option<&str>,
        nr: Option<i64>,
    ) -> Result<&Molecule, TngError> {
        let n_molecules = self.n_molecules;

        for i in (0..n_molecules).rev() {
            let molecule = &self.molecules[i as usize];
            if (name.is_none() || name.unwrap() == molecule.name)
                && (nr.is_none() || nr.unwrap() == molecule.id)
            {
                return Ok(molecule);
            }
        }

        Err(TngError::NotFound("molecule not found".to_string()))
    }

    /// C API: `tng_molecule_name_get`.
    pub(crate) fn molecule_name_get<'a>(
        &'a self,
        molecule: &'a Molecule,
        max_len: usize,
    ) -> Result<&'a str, TngError> {
        Self::validate_get_name_len(&molecule.name, "molecule name", max_len)
    }

    /// C API: `tng_molecule_num_chains_get`.
    pub(crate) fn molecule_num_chains_get(&self, molecule: &Molecule) -> i64 {
        molecule.n_chains
    }

    /// C API: `tng_molecule_chain_of_index_get`.
    ///
    /// Retrieve the [`crate::chain::Chain`] of a molecule with specified index in the list of chains.
    ///
    /// # Errors
    ///
    /// Returns [`TngError::NotFound`] if `index` is bigger than the amount of chains
    pub(crate) fn molecule_chain_of_index_get<'a>(
        &'a self,
        molecule: &'a Molecule,
        index: usize,
    ) -> Result<&'a Chain, TngError> {
        if index >= molecule.n_chains as usize {
            return Err(TngError::NotFound(format!(
                "chain index was bigger than the amount of chains. index: {index}"
            )));
        }

        Ok(&molecule.chains[index])
    }

    /// Get the number of [`crate::residue::Residue`] in the molecule.
    ///
    /// C API: `tng_molecule_num_residues_get`.
    pub(crate) fn molecule_num_residues_get(&self, molecule: &Molecule) -> i64 {
        molecule.n_residues
    }

    /// Retrieve the [`crate::residue::Residue`] of a molecule with specified index in the list of residues.
    ///
    /// C API: `tng_molecule_residue_of_index_get`.
    ///
    /// # Errors
    ///
    /// Returns [`TngError::NotFound`] if `index` is bigger than the amount of residues
    pub(crate) fn molecule_residue_of_index_get<'a>(
        &'a self,
        molecule: &'a Molecule,
        index: usize,
    ) -> Result<&'a Residue, TngError> {
        if index >= molecule.n_residues as usize {
            return Err(TngError::NotFound(format!(
                "residue index was bigger than the amount of residues. Index: {index}"
            )));
        }

        Ok(&molecule.residues[index])
    }

    /// C API: `tng_molecule_num_atoms_get`.
    pub(crate) fn molecule_num_atoms_get(&self, molecule: &Molecule) -> i64 {
        molecule.n_atoms
    }

    /// C API: `tng_molecule_atom_of_index_get`.
    pub(crate) fn molecule_atom_of_index_get<'a>(
        &'a self,
        molecule: &'a Molecule,
        index: usize,
    ) -> Result<&'a Atom, TngError> {
        if index >= molecule.n_atoms as usize {
            return Err(TngError::NotFound(format!(
                "atom index was bigger than the amount of atoms. Index: {index}"
            )));
        }

        Ok(&molecule.atoms[index])
    }

    /// C API: `tng_chain_name_get`.
    pub(crate) fn chain_name_get<'a>(
        &'a self,
        chain: &'a Chain,
        max_len: usize,
    ) -> Result<&'a str, TngError> {
        Self::validate_get_name_len(&chain.name, "chain name", max_len)
    }

    /// C API: `tng_chain_num_residues_get`.
    ///
    /// Get the number of residues in a molecule chain.
    pub(crate) fn chain_num_residues_get(&self, chain: &Chain) -> u64 {
        chain.n_residues
    }

    /// C API: `tng_chain_residue_of_index_get`.
    ///
    /// Retrieve the residue of a chain with specified index in the list of residues.
    pub(crate) fn chain_residue_of_index_get<'a>(
        &'a self,
        chain: &Chain,
        index: usize,
    ) -> Result<&'a Residue, TngError> {
        if index >= chain.n_residues as usize {
            return Err(TngError::NotFound(format!(
                "A residue with index {index} was not found in the chain."
            )));
        }

        let molecule = &self.molecules[chain.parent_molecule_idx];
        let residue_index = chain.residues_indices.0 + index;
        Ok(&molecule.residues[residue_index])
    }

    /// C API: `tng_chain_residue_find`.
    pub(crate) fn chain_residue_find(
        &self,
        chain: &Chain,
        name: Option<&str>,
        id: Option<usize>,
    ) -> Result<&Residue, TngError> {
        let molecule = &self.molecules[chain.parent_molecule_idx];
        let n_residue = chain.n_residues;

        for i in (0..n_residue as usize).rev() {
            let residue = &molecule.residues[chain.residues_indices.0 + i];

            if (name.is_none() || name.unwrap() == residue.name)
                && (id.is_none() || id.unwrap() as u64 == residue.id)
            {
                return Ok(residue);
            }
        }

        Err(TngError::NotFound("residue not found".to_string()))
    }

    /// C API: `tng_residue_name_get`.
    pub(crate) fn residue_name_get<'a>(
        &'a self,
        residue: &'a Residue,
        max_len: usize,
    ) -> Result<&'a str, TngError> {
        Self::validate_get_name_len(&residue.name, "residue name", max_len)
    }

    /// C API: `tng_residue_num_atoms_get`.
    pub(crate) fn residue_num_atoms_get(&self, residue: &Residue) -> u64 {
        residue.n_atoms
    }

    /// C API: `tng_residue_atom_of_index_get`.
    pub(crate) fn residue_atom_of_index_get<'a>(
        &'a self,
        residue: &Residue,
        index: usize,
    ) -> Result<&'a Atom, TngError> {
        if index >= residue.n_atoms as usize {
            return Err(TngError::NotFound(format!(
                "An atom with index {index} was not found in the residue."
            )));
        }

        let molecule = &self.molecules[residue.parent_molecule_idx];
        Ok(&molecule.atoms[residue.atoms_offset + index])
    }

    /// C API: `tng_atom_residue_get`.
    pub(crate) fn atom_residue_get<'a>(&'a self, atom: &Atom) -> Result<&'a Residue, TngError> {
        let residue_index = atom
            .residue_index
            .ok_or_else(|| TngError::NotFound("atom is not part of a residue".to_string()))?;

        let molecule = &self.molecules[atom.parent_molecule_idx];
        molecule
            .residues
            .get(residue_index)
            .ok_or_else(|| TngError::NotFound("residue not found".to_string()))
    }

    /// C API: `tng_atom_name_get`.
    pub(crate) fn atom_name_get<'a>(
        &'a self,
        atom: &'a Atom,
        max_len: usize,
    ) -> Result<&'a str, TngError> {
        Self::validate_get_name_len(&atom.name, "atom name", max_len)
    }

    /// C API: `tng_atom_type_get`.
    pub(crate) fn atom_type_get<'a>(
        &'a self,
        atom: &'a Atom,
        max_len: usize,
    ) -> Result<&'a str, TngError> {
        Self::validate_get_name_len(&atom.atom_type, "atom type", max_len)
    }

    fn gen_data_get(
        &mut self,
        is_particle_data: bool,
        block_id: BlockID,
    ) -> Option<(i64, i64, i64, DataType, Vec<f64>)> {
        self.gen_data_vector_get(is_particle_data, block_id)
    }

    pub(crate) fn util_time_of_frame_get(&mut self, frame_nr: i64) -> Result<f64, TngError> {
        self.frame_set_of_frame_find(frame_nr)?;

        let frame_set = &self.current_trajectory_frame_set;

        if self.time_per_frame <= 0.0 {
            return Err(TngError::Constraint(format!(
                "time_per_frame was <= 0. Got {}",
                self.time_per_frame
            )));
        }

        Ok(frame_set.first_frame_time
            + (self.time_per_frame * (frame_nr - frame_set.first_frame) as f64))
    }

    pub(crate) fn util_num_frames_with_data_of_block_id_get(
        &mut self,
        block_id: BlockID,
    ) -> Result<i64, TngError> {
        let mut n_frames = 0;
        self.input_file_init();

        let first_frame_set_file_pos = self.first_trajectory_frame_set_input_file_pos;
        let curr_file_pos = self.get_input_file_position();
        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(first_frame_set_file_pos as u64))
            .map_err(|e| {
                TngError::Critical(format!(
                    "Cannot seek to position {first_frame_set_file_pos} in block_header_read: {e}"
                ))
            })?;

        let mut stat = self.frame_set_n_frames_of_data_block_get(block_id);

        while stat.is_ok() && self.current_trajectory_frame_set.next_frame_set_file_pos != -1 {
            n_frames += stat.unwrap();
            let next_pos = self.current_trajectory_frame_set.next_frame_set_file_pos as u64;
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(next_pos))
                .map_err(|e| {
                    TngError::Critical(format!(
                        "Cannot seek to position {next_pos} in util_num_frames_with_data_of_block_id_get: {e}"
                    ))
                })?;
            stat = self.frame_set_n_frames_of_data_block_get(block_id);
        }
        if let Ok(curr_n_frames) = stat {
            n_frames += curr_n_frames;
        }
        self.input_file
            .as_ref()
            .expect("init input_file")
            .seek(SeekFrom::Start(curr_file_pos))
            .map_err(|e| {
                TngError::Critical(format!(
                    "Cannot seek to position {curr_file_pos} in block_header_read: {e}"
                ))
            })?;

        Ok(n_frames)
    }

    fn frame_set_n_frames_of_data_block_get(&mut self, block_id: BlockID) -> Result<i64, TngError> {
        let mut found = false;
        let mut block = GenBlock::new();

        let stat = self.block_header_read(&mut block);
        // If the block header could not be read the reading position might not have been
        // at the start of a block. Try again from the file position of the current frame
        // set.
        if stat.is_err() {
            let file_pos = self.current_trajectory_frame_set_input_file_pos as u64;
            self.input_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(file_pos))
                .map_err(|e| {
                    TngError::Critical(format!(
                        "Cannot seek to position {file_pos} in block_header_read: {e}"
                    ))
                })?;
            self.block_header_read(&mut block)?;
        }

        if block.id == BlockID::TrajectoryFrameSet {
            self.block_read_next(&mut block, SKIP_HASH);
            self.block_header_read(&mut block)?;
        }

        let mut metainfo = None;
        while block.id != BlockID::TrajectoryFrameSet && !found {
            if block.id == block_id {
                let temp = self.data_block_meta_information_read(&mut block);
                metainfo = Some(temp);
                found = true;
            } else {
                self.input_file
                    .as_ref()
                    .expect("init input_file")
                    .seek(SeekFrom::Current(block.block_contents_size as i64))
                    .map_err(|e| {
                        TngError::Critical(format!(
                            "Cannot seek to position {} in block_header_read: {e}",
                            block.block_contents_size
                        ))
                    })?;
                self.block_header_read(&mut block)?;
            }
        }
        let n_frames = if found {
            let metainfo = metainfo.expect("we found a block");
            (self.current_trajectory_frame_set.n_frames
                - (self.current_trajectory_frame_set.first_frame - metainfo.first_frame_with_data))
                / metainfo.stride_length
        } else {
            0
        };

        Ok(n_frames)
    }

    pub fn util_pos_read_range(
        &mut self,
        first_frame: i64,
        last_frame: i64,
    ) -> Result<(Vec<f64>, i64), TngError> {
        assert!(
            first_frame <= last_frame,
            "`first_frame`, must be lower or equal to `last_frame`"
        );

        let (positions, _n_particles, _n_values_per_frame, stride_length, data_type) = self
            .particle_data_vector_interval_get(
                BlockID::TrajPositions,
                first_frame,
                last_frame,
                USE_HASH,
            )?;
        if data_type != DataType::Float {
            return Err(TngError::Constraint("data was not float".to_string()));
        }

        Ok((positions, stride_length))
    }

    pub(crate) fn util_trajectory_next_frame_present_data_blocks_find(
        &mut self,
        current_frame: i64,
        n_requested_data_block_ids: i64,
        requested_data_block_ids: &[BlockID],
    ) -> Result<(i64, i64), TngError> {
        let mut data;
        let mut read_all = false;
        let mut min_diff;
        let frame_set_file_pos;
        let mut current_frame = current_frame;
        let mut data_block_ids_in_next_frame =
            vec![BlockID::default(); n_requested_data_block_ids as usize];
        let mut n_data_blocks_in_next_frame = 0;

        current_frame += 1;

        if current_frame < self.current_trajectory_frame_set.first_frame
            || current_frame
                >= self.current_trajectory_frame_set.first_frame
                    + self.current_trajectory_frame_set.n_frames
        {
            frame_set_file_pos = self.current_trajectory_frame_set_input_file_pos;
            let stat = self.frame_set_of_frame_find(current_frame);
            if let Err(err) = stat {
                // If the frame set search found the frame set after the starting
                // frame set there is a gap in the frame sets. So, even if the frame
                // was not found the next frame with data is still in the found
                // frame set.
                if matches!(err, TngError::Critical(_))
                    || self.current_trajectory_frame_set.prev_frame_set_file_pos
                        != frame_set_file_pos
                {
                    return Err(err);
                }
                current_frame = self.current_trajectory_frame_set.first_frame;
            }
        }

        // Check for data blocks only if they have not already been found
        if self.current_trajectory_frame_set.n_particle_data_blocks <= 0
            && self.current_trajectory_frame_set.n_data_blocks <= 0
        {
            let mut file_pos = self.get_input_file_position();
            if file_pos < self.input_file_len {
                let mut block = GenBlock::new();
                self.block_header_read(&mut block)?;
                while file_pos < self.input_file_len
                    && block.id != BlockID::TrajectoryFrameSet
                    && block.id != BlockID::Unknown
                {
                    self.block_read_next(&mut block, USE_HASH);
                    file_pos = self.get_input_file_position();
                    if file_pos < self.input_file_len {
                        self.block_header_read(&mut block)?;
                    }
                }
            }
            read_all = true;
        }

        min_diff = -1;

        for i in 0..self.current_trajectory_frame_set.n_particle_data_blocks {
            let data = &self.current_trajectory_frame_set.tr_particle_data[i].clone();
            let block_id = data.block_id;

            if n_requested_data_block_ids > 0 {
                let mut found = false;
                for item in requested_data_block_ids
                    .iter()
                    .take(n_requested_data_block_ids as usize)
                {
                    if block_id == *item {
                        found = true;
                        break;
                    }
                }
                if !found {
                    continue;
                }
            }

            if !read_all
                && data.last_retrieved_frame < self.current_trajectory_frame_set.first_frame
                || data.last_retrieved_frame
                    >= self.current_trajectory_frame_set.first_frame
                        + self.current_trajectory_frame_set.n_frames
            {
                let stat = self.frame_set_read_current_only_data_from_block_id(USE_HASH, block_id);
                match stat {
                    Ok(_) => {}
                    Err(TngError::Constraint(_)) | Err(TngError::NotFound(_)) => continue,
                    _ => {
                        return Err(TngError::Critical(format!(
                            "Cannot read data block of frame set. {}:{}",
                            file!(),
                            line!()
                        )));
                    }
                }
            }
            let data_frame = if self.current_trajectory_frame_set.first_frame != current_frame
                && data.last_retrieved_frame >= 0
            {
                data.last_retrieved_frame + data.stride_length
            } else {
                data.first_frame_with_data
            };
            let frame_diff = data_frame - current_frame;
            if frame_diff < 0 {
                continue;
            }
            if min_diff == -1 || frame_diff <= min_diff {
                n_data_blocks_in_next_frame = if frame_diff < min_diff {
                    1
                } else {
                    n_data_blocks_in_next_frame + 1
                };
                if n_requested_data_block_ids <= 0 {
                    data_block_ids_in_next_frame =
                        vec![BlockID::default(); n_data_blocks_in_next_frame as usize];
                } else {
                    assert!(
                        n_data_blocks_in_next_frame <= n_requested_data_block_ids,
                        "Array of data block IDs out of bounds"
                    );
                }
                *data_block_ids_in_next_frame
                    .last_mut()
                    .expect("we've just allocated this array") = block_id;

                min_diff = frame_diff;
            }
        }

        for i in 0..self.current_trajectory_frame_set.n_data_blocks {
            data = self.current_trajectory_frame_set.tr_data[i].clone();
            let block_id = data.block_id;

            if n_requested_data_block_ids > 0 {
                let mut found = false;
                for item in requested_data_block_ids
                    .iter()
                    .take(n_requested_data_block_ids as usize)
                {
                    if block_id == *item {
                        found = true;
                        break;
                    }
                }
                if !found {
                    continue;
                }
            }

            if !read_all
                && (data.last_retrieved_frame < self.current_trajectory_frame_set.first_frame
                    || data.last_retrieved_frame
                        >= self.current_trajectory_frame_set.first_frame
                            + self.current_trajectory_frame_set.n_frames)
            {
                let stat = self.frame_set_read_current_only_data_from_block_id(USE_HASH, block_id);
                match stat {
                    Ok(_) => {}
                    Err(TngError::Constraint(_)) | Err(TngError::NotFound(_)) => continue,
                    _ => {
                        return Err(TngError::Critical(format!(
                            "Cannot read data block of frame set. {}:{}",
                            file!(),
                            line!()
                        )));
                    }
                }
            }
            let data_frame = if self.current_trajectory_frame_set.first_frame != current_frame
                && data.last_retrieved_frame >= 0
            {
                data.last_retrieved_frame + data.stride_length
            } else {
                data.first_frame_with_data
            };
            let frame_diff = data_frame - current_frame;
            if frame_diff < 0 {
                continue;
            }
            if min_diff == -1 || frame_diff <= min_diff {
                n_data_blocks_in_next_frame = if frame_diff < min_diff {
                    1
                } else {
                    n_data_blocks_in_next_frame + 1
                };
                if n_requested_data_block_ids <= 0 {
                    data_block_ids_in_next_frame =
                        vec![BlockID::default(); n_data_blocks_in_next_frame as usize];
                } else {
                    assert!(
                        n_data_blocks_in_next_frame <= n_requested_data_block_ids,
                        "Array of data block IDs out of bounds"
                    );
                }
                *data_block_ids_in_next_frame
                    .last_mut()
                    .expect("we've just allocated this array") = block_id;

                min_diff = frame_diff;
            }
        }
        if min_diff < 0 {
            return Err(TngError::Constraint(format!(
                "`min_diff` was negative. Got {min_diff}"
            )));
        }

        let next_frame = current_frame + min_diff;
        Ok((next_frame, n_data_blocks_in_next_frame))
    }
    pub(crate) fn util_frame_current_compression_get(
        &mut self,
        block_id: BlockID,
    ) -> Result<(Compression, f64), TngError> {
        let block_type;
        let mut data;
        let mut i = None;
        data = self.particle_data_find(block_id);

        if data.is_some() {
            block_type = Some(ParticleDependency::ParticleBlockData);
        } else {
            data = self.data_find(block_id);
            if data.is_some() {
                block_type = Some(ParticleDependency::NonParticleBlockData);
            } else {
                self.frame_set_read_current_only_data_from_block_id(USE_HASH, block_id)?;
                data = self.particle_data_find(block_id);
                if data.is_some() {
                    block_type = Some(ParticleDependency::ParticleBlockData);
                } else {
                    data = self.data_find(block_id);
                    if data.is_some() {
                        block_type = Some(ParticleDependency::NonParticleBlockData);
                    } else {
                        return Err(TngError::NotFound("Could not find data".to_string()));
                    }
                }
            }
        }
        // if (block_type == TNG_PARTICLE_BLOCK_DATA || block_type == TNG_NON_PARTICLE_BLOCK_DATA)
        if block_type.is_some() {
            if let Some(data) = data.as_ref() {
                i = if data.last_retrieved_frame < 0 {
                    Some(data.first_frame_with_data)
                } else {
                    Some(data.last_retrieved_frame)
                }
            }
        } else {
            return Err(TngError::NotFound(
                "Could not find an appropriate data block".to_string(),
            ));
        }
        if let Some(i) = i
            && (i < self.current_trajectory_frame_set.first_frame
                || i >= self.current_trajectory_frame_set.first_frame
                    + self.current_trajectory_frame_set.n_frames)
        {
            self.frame_set_of_frame_find(i)?;
            self.frame_set_read_current_only_data_from_block_id(USE_HASH, block_id)?;
        }
        // if (block_type == TNG_PARTICLE_BLOCK_DATA || block_type == TNG_NON_PARTICLE_BLOCK_DATA)
        if block_type.is_some()
            && let Some(data) = data
        {
            let codec_id = data.codec_id;
            let factor = data.compression_multiplier;
            return Ok((codec_id, factor));
        }

        unreachable!("we should've found a block type OR errored out")
    }

    pub(crate) fn util_trajectory_close(&mut self) -> Result<(), TngError> {
        if self.current_trajectory_frame_set.n_unwritten_frames > 0 {
            self.current_trajectory_frame_set.n_frames =
                self.current_trajectory_frame_set.n_unwritten_frames;
            self.frame_set_write(USE_HASH)?;
        }

        self.trajectory_destroy()?;
        Ok(())
    }

    /// C API: `tng_trajectory_destroy`.
    pub fn trajectory_destroy(&mut self) -> Result<(), TngError> {
        let same_file = match (self.input_file.as_ref(), self.output_file.as_ref()) {
            (Some(input), Some(output)) => is_same_file(input, output)?,
            _ => false,
        };

        if self.input_file.is_some() {
            if same_file {
                self.frame_set_finalize(USE_HASH)?;
                self.output_file = None;
            }

            drop(self.input_file.take());
        }

        self.input_file_path = None;

        if self.output_file.is_some() {
            let _ = self.frame_set_finalize(USE_HASH);
            drop(self.output_file.take());
        }

        self.output_file_path = None;

        *self = Trajectory::new();

        Ok(())
    }

    /// C API: tng_trajectory_init_from_src
    ///
    /// Copy a trajectory data container (dest is setup as well)
    pub(crate) fn trajectory_init_from_src(&self) -> Self {
        let mut dest = Trajectory::new();
        let frame_set = &mut dest.current_trajectory_frame_set;

        if let Some(input_file_path) = self.input_file_path.as_ref() {
            dest.input_file_path = Some((*input_file_path).clone());
            dest.input_file_len = self.input_file_len;
        } else {
            dest.input_file_path = None;
        }
        dest.input_file = None;
        if let Some(output_file_path) = self.output_file_path.as_ref() {
            dest.output_file_path = Some((*output_file_path).clone());
        } else {
            dest.output_file_path = None;
        }
        dest.output_file = None;

        dest.first_program_name = String::new();
        dest.first_user_name = String::new();
        dest.first_computer_name = String::new();
        dest.first_pgp_signature = String::new();
        dest.last_program_name = String::new();
        dest.last_user_name = String::new();
        dest.last_computer_name = String::new();
        dest.last_pgp_signature = String::new();
        dest.forcefield_name = String::new();

        dest.var_num_atoms = self.var_num_atoms;
        dest.first_trajectory_frame_set_input_file_pos =
            self.first_trajectory_frame_set_input_file_pos;
        dest.last_trajectory_frame_set_input_file_pos =
            self.last_trajectory_frame_set_input_file_pos;
        dest.current_trajectory_frame_set_input_file_pos =
            self.current_trajectory_frame_set_input_file_pos;
        dest.first_trajectory_frame_set_output_file_pos =
            self.first_trajectory_frame_set_output_file_pos;
        dest.last_trajectory_frame_set_output_file_pos =
            self.last_trajectory_frame_set_output_file_pos;
        dest.current_trajectory_frame_set_output_file_pos =
            self.current_trajectory_frame_set_output_file_pos;
        dest.frame_set_n_frames = self.frame_set_n_frames;
        dest.n_trajectory_frame_sets = self.n_trajectory_frame_sets;
        dest.medium_stride_length = self.medium_stride_length;
        dest.long_stride_length = self.long_stride_length;

        dest.time_per_frame = self.time_per_frame;

        // Currently the non trajectory data blocks are not copied since it
        // can lead to problems when freeing memory in a parallel block.
        dest.n_particle_data_blocks = 0;
        dest.n_data_blocks = 0;
        dest.non_tr_particle_data = Vec::new();
        dest.non_tr_data = Vec::new();

        dest.compress_algo_pos = Vec::new();
        dest.compress_algo_vel = Vec::new();
        dest.distance_unit_exponential = -9;
        dest.compression_precision = 1000.0;

        frame_set.n_mapping_blocks = 0;
        frame_set.mappings = Vec::new();
        frame_set.molecule_cnt_list = Vec::new();

        frame_set.n_particle_data_blocks = 0;
        frame_set.n_data_blocks = 0;

        frame_set.tr_particle_data = Vec::new();
        frame_set.tr_data = Vec::new();

        frame_set.n_written_frames = 0;
        frame_set.n_unwritten_frames = 0;

        frame_set.next_frame_set_file_pos = -1;
        frame_set.prev_frame_set_file_pos = -1;
        frame_set.medium_stride_next_frame_set_file_pos = -1;
        frame_set.medium_stride_prev_frame_set_file_pos = -1;
        frame_set.long_stride_next_frame_set_file_pos = -1;
        frame_set.long_stride_prev_frame_set_file_pos = -1;
        frame_set.first_frame = -1;

        dest.n_molecules = 0;
        dest.molecules = Vec::new();
        dest.molecule_cnt_list = Vec::new();
        dest.n_particles = self.n_particles;

        dest.endianness32 = self.endianness32;
        dest.endianness64 = self.endianness64;
        dest.input_swap32 = self.input_swap32;
        dest.input_swap64 = self.input_swap64;
        dest.output_swap32 = self.output_swap32;
        dest.output_swap64 = self.output_swap64;

        dest.current_trajectory_frame_set.next_frame_set_file_pos = -1;
        dest.current_trajectory_frame_set.prev_frame_set_file_pos = -1;
        dest.current_trajectory_frame_set.n_frames = 0;

        dest
    }

    fn frame_set_finalize(&mut self, _use_hash: bool) -> Result<(), TngError> {
        if self.current_trajectory_frame_set.n_written_frames
            == self.current_trajectory_frame_set.n_frames
        {
            return Ok(());
        }

        self.current_trajectory_frame_set.n_written_frames =
            self.current_trajectory_frame_set.n_frames;

        self.output_file_init();

        let temp = self.input_file.take();
        let result = (|| -> Result<(), TngError> {
            let mut block = GenBlock::new();
            self.input_file = Some(
                self.output_file
                    .as_ref()
                    .expect("we just created the output_file")
                    .try_clone()?,
            );
            let curr_file_pos = self.get_output_file_position();
            let pos = self.current_trajectory_frame_set_output_file_pos;

            self.output_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(pos as u64))
                .map_err(|e| {
                    TngError::Critical(format!(
                        "Cannot seek to position {pos} in frame_set_initialize: {e}"
                    ))
                })?;
            self.block_header_read(&mut block)?;

            self.output_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Current(
                    size_of_val(&self.current_trajectory_frame_set.first_frame) as i64,
                ))
                .map_err(|e| {
                    TngError::Critical(format!(
                        "Cannot seek 8 bytes forward in frame_set_initialize: {e}"
                    ))
                })?;
            let out_file = self.output_file.as_mut().expect("init output_file");
            out_file
                .write_all(
                    &(size_of_val(&self.current_trajectory_frame_set.first_frame) as i64)
                        .to_ne_bytes(),
                )
                .expect("able to write to output_file");

            // TODO: hash mode tng_io.c line 6242

            self.output_file
                .as_ref()
                .expect("init input_file")
                .seek(SeekFrom::Start(curr_file_pos))
                .map_err(|e| {
                    TngError::Critical(format!(
                        "Cannot seek to position {curr_file_pos} in frame_set_initialize: {e}"
                    ))
                })?;

            Ok(())
        })();

        self.input_file = temp;
        result
    }

    /// C API: tng_util_vel_with_time_double_write
    ///
    /// High-level function for adding data to velocities data blocks at
    /// double precision. If the frame is at the beginning of a frame set the
    /// time stamp of the frame set is set.
    pub(crate) fn util_vel_with_time_double_write(
        &mut self,
        frame_nr: i64,
        time: f64,
        velocities: &mut [f64],
    ) -> Result<(), TngError> {
        self.util_generic_with_time_double_write(
            frame_nr,
            time,
            velocities,
            3,
            BlockID::TrajVelocities,
            "VELOCITIES",
            ParticleDependency::ParticleBlockData,
            Compression::TNG,
        )
    }

    fn util_generic_with_time_double_write(
        &mut self,
        frame_nr: i64,
        time: f64,
        values: &mut [f64],
        n_values_per_frame: i64,
        block_id: BlockID,
        block_name: &str,
        particle_dependency: ParticleDependency,
        compression: Compression,
    ) -> Result<(), TngError> {
        self.util_generic_double_write(
            frame_nr,
            values,
            n_values_per_frame,
            block_id,
            block_name,
            particle_dependency,
            compression,
        )
    }

    fn util_generic_double_write(
        &mut self,
        frame_nr: i64,
        values: &mut [f64],
        n_values_per_frame: i64,
        block_id: BlockID,
        block_name: &str,
        particle_dependency: ParticleDependency,
        compression: Compression,
    ) -> Result<(), TngError> {
        let mut data;
        let mut n_particles = 0;
        let frame_pos;
        let block_type_flag;
        let mut stride_length = 100;
        let mut last_frame;
        let mut is_first_frame_flag = false;
        let n_frames;
        if particle_dependency == ParticleDependency::ParticleBlockData {
            n_particles = self.num_particles_get();
            assert!(
                n_particles > 0,
                "There must be particles in the system to write particle data"
            );
        }

        if frame_nr < 0 {
            block_type_flag = BlockType::NonTrajectory;
            n_frames = 1;
            stride_length = 1;
        } else {
            block_type_flag = BlockType::Trajectory;

            // TODO: do we need to check !frame_set here like the C code does? tng_io.c line 15717
            if self.n_trajectory_frame_sets <= 0 {
                self.frame_set_new(0, self.frame_set_n_frames)?;
            }
            last_frame = self.current_trajectory_frame_set.first_frame
                + self.current_trajectory_frame_set.n_frames
                - 1;
            if frame_nr > last_frame {
                self.frame_set_write(USE_HASH)?;
                if last_frame + self.frame_set_n_frames < frame_nr {
                    last_frame = frame_nr + 1;
                }
                self.frame_set_new(last_frame + 1, self.frame_set_n_frames)?;
            }
            if self.current_trajectory_frame_set.n_unwritten_frames == 0 {
                is_first_frame_flag = true;
            }
            self.current_trajectory_frame_set.n_unwritten_frames =
                frame_nr - self.current_trajectory_frame_set.first_frame + 1;

            n_frames = self.current_trajectory_frame_set.n_frames;
        }

        if particle_dependency == ParticleDependency::ParticleBlockData {
            data = self.particle_data_find(block_id);
            if data.is_none() {
                self.particle_data_block_add(
                    block_id,
                    block_name,
                    DataType::Double,
                    &block_type_flag,
                    n_frames,
                    n_values_per_frame,
                    stride_length,
                    0,
                    n_particles,
                    compression,
                    None,
                )?;
                data = if block_type_flag == BlockType::Trajectory {
                    Some(
                        self.current_trajectory_frame_set.tr_particle_data
                            [self.current_trajectory_frame_set.n_particle_data_blocks - 1]
                            .clone(),
                    )
                } else {
                    Some(self.non_tr_particle_data[self.n_particle_data_blocks - 1].clone())
                };
                data.as_mut()
                    .expect("we just filled it with Some")
                    .allocate_particle_data_mem(
                        n_frames,
                        stride_length,
                        n_particles,
                        n_values_per_frame,
                    );
            }
            // (C)FIXME: Here we must be able to handle modified n_particles as well
            else if n_frames > data.as_ref().expect("has to be Some").n_frames {
                let data = data.as_mut().unwrap();
                data.allocate_particle_data_mem(
                    n_frames,
                    stride_length,
                    n_particles,
                    n_values_per_frame,
                );
            }

            let mut data = data.unwrap();
            if block_type_flag == BlockType::Trajectory {
                stride_length = data.stride_length;

                if is_first_frame_flag
                    || data.first_frame_with_data < self.current_trajectory_frame_set.first_frame
                {
                    data.first_frame_with_data = frame_nr;
                    frame_pos = 0;
                } else {
                    frame_pos =
                        (frame_nr - self.current_trajectory_frame_set.first_frame) / stride_length;
                }

                data.values.as_mut().expect("values to be allocated")[((frame_pos
                    * n_particles
                    * n_values_per_frame)
                    as usize
                    * size_of::<f64>())..]
                    .copy_from_slice(
                        &values[..(n_particles * n_values_per_frame) as usize]
                            .iter()
                            .flat_map(|&x| x.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    );
            } else {
                data.values
                    .as_mut()
                    .expect("values to be allocated")
                    .copy_from_slice(
                        &values[..(n_particles * n_values_per_frame) as usize]
                            .iter()
                            .flat_map(|&x| x.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    );
            }
        } else {
            // unimplemented!("")
        }

        Ok(())
    }

    /// C API: tng_molecule_system_copy
    ///
    /// Copy all molecules and the molecule counts from one TNG trajectory ([`Self`]) to another [`traj_dest`]
    pub(crate) fn molecule_system_copy(&self, traj_dest: &mut Trajectory) {
        traj_dest.n_molecules = 0;
        traj_dest.n_particles = 0;
        traj_dest.molecules = Vec::with_capacity(self.n_molecules as usize);
        traj_dest.molecule_cnt_list = Vec::with_capacity(self.n_molecules as usize);

        for i in 0..self.n_molecules as usize {
            let molecule = self.molecules[i].clone();
            traj_dest.molecules.push(molecule);
            traj_dest.molecule_cnt_list.push(0);
            traj_dest.n_molecules += 1;
            traj_dest.molecule_cnt_set(i, self.molecule_cnt_list[i]);
        }
    }

    fn get_property(&mut self, block_id: BlockID) -> Result<(Vec<f32>, i64), TngError> {
        let n_frames = self.num_frames_get().expect("there has to be Some frames");
        let (property, _n_particles, _n_values_per_frame, stride_length, data_type) =
            self.particle_data_vector_interval_get(block_id, 0, n_frames - 1, USE_HASH)?;
        if data_type != DataType::Float {
            return Err(TngError::Constraint("data was not float".to_string()));
        }
        Ok((
            property.into_iter().map(|x| x as f32).collect(),
            stride_length,
        ))
    }

    pub fn util_pos_read(&mut self) -> Result<(Vec<f32>, i64), TngError> {
        self.get_property(BlockID::TrajPositions)
    }

    pub fn util_vel_read(&mut self) -> Result<(Vec<f32>, i64), TngError> {
        self.get_property(BlockID::TrajVelocities)
    }

    pub fn util_force_read(&mut self) -> Result<(Vec<f32>, i64), TngError> {
        self.get_property(BlockID::TrajForces)
    }

    pub fn util_box_shape_read(&mut self) -> Result<(Vec<f32>, i64), TngError> {
        let n_frames = self.num_frames_get().expect("there has to be Some frames");
        let (values, _n_particles, _n_values_per_frame, stride_length, data_type) = self
            .gen_data_vector_interval_get(
                BlockID::TrajBoxShape,
                false,
                0,
                n_frames - 1,
                USE_HASH,
            )?;
        if data_type != DataType::Float {
            return Err(TngError::Constraint(format!(
                "data type was not Float, but {data_type:?}"
            )));
        }

        Ok((
            values.into_iter().map(|x| x as f32).collect(),
            stride_length,
        ))
    }
}

/// Interpret a byte slice as a numeric value and return it as f64.
/// The slice must be exactly `data_type.get_size()` bytes.
fn interval_bytes_to_f64(bytes: &[u8], data_type: DataType) -> f64 {
    match data_type {
        DataType::Int => {
            let arr = <[u8; 8]>::try_from(bytes).expect("8 bytes for i64");
            i64::from_ne_bytes(arr) as f64
        }
        DataType::Float => {
            let arr = <[u8; 4]>::try_from(bytes).expect("4 bytes for f32");
            f32::from_ne_bytes(arr) as f64
        }
        DataType::Double => {
            let arr = <[u8; 8]>::try_from(bytes).expect("8 bytes for f64");
            f64::from_ne_bytes(arr)
        }
        DataType::Char => unreachable!("char handled separately"),
    }
}

/// Uncompresses any tng compress block, positions or velocities. It determines whether it is
/// positions or velocities from the data buffer. The return value is 0 if ok, and 1 if not.
pub(crate) fn compress_uncompress<T: Float>(data: &[u8], posvel: &mut [T]) -> Result<(), TngError> {
    let magic_int = u32::from(readbufferfix(data, 4));

    match magic_int {
        MAGIC_INT_POS => compress_uncompress_pos(data, posvel)?,
        MAGIC_INT_VEL => compress_uncompress_vel(data, posvel)?,
        _ => {
            return Err(TngError::Constraint(format!(
                "found the wrong magic int when decompressing. Found {magic_int}"
            )));
        }
    };

    Ok(())
}

/// Uncompresses any tng compress block, positions or velocities. It determines whether it is
/// positions or velocities from the data buffer. The return value is 0 if ok, and 1 if not.
pub(crate) fn compress_uncompress_int(
    data: &[u8],
    posvel: &mut [i32],
) -> Result<(FixT, FixT), TngError> {
    let magic_int = u32::from(readbufferfix(data, 4));

    match magic_int {
        MAGIC_INT_POS => compress_uncompress_pos_int(data, posvel),
        MAGIC_INT_VEL => compress_uncompress_vel_int(data, posvel),
        _ => Err(TngError::Constraint(format!(
            "found the wrong magic int when decompressing. Found {magic_int}"
        ))),
    }
}

pub(crate) fn compress_int_to_real<T: Float>(
    posvel_int: &[i32],
    prec_hi: FixT,
    prec_lo: FixT,
    n_atoms: usize,
    n_frames: usize,
    posvel_real: &mut [T],
) {
    unquantize(
        posvel_real,
        n_atoms,
        n_frames,
        T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
        posvel_int,
    );
}

fn compress_uncompress_pos<T: Float>(data: &[u8], pos: &mut [T]) -> Result<(), TngError> {
    let (_prec_hi, _prec_lo) = compress_uncompress_pos_gen(data, Some(pos), None)?;
    Ok(())
}

fn compress_uncompress_pos_int(data: &[u8], pos: &mut [i32]) -> Result<(FixT, FixT), TngError> {
    compress_uncompress_pos_gen::<f64>(data, None, Some(pos))
}

fn compress_uncompress_pos_gen<T: Float>(
    data: &[u8],
    mut posdf: Option<&mut [T]>,
    mut posi: Option<&mut [i32]>,
) -> Result<(FixT, FixT), TngError> {
    let mut bufloc = 0;

    // Magic integer for positions
    let magic_int = u32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    if magic_int != MAGIC_INT_POS {
        return Err(TngError::Constraint(format!(
            "Expected MAGIC_INT_POS, got {magic_int}"
        )));
    }

    // Number of atoms
    let natoms = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Number of frames
    let nframes = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Initial coding
    let initial_coding = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Initial coding parameter
    let initial_coding_parameter = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Coding
    let coding = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Coding parameter.
    let coding_parameter = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Precision
    let prec_lo = readbufferfix(&data[bufloc..], 4);
    bufloc += 4;
    let prec_hi = readbufferfix(&data[bufloc..], 4);
    bufloc += 4;

    // Allocate the memory for the quantized positions
    let mut quant = vec![0; (natoms * nframes) as usize * 3];
    // The data block length
    let length = readbufferfix(&data[bufloc..], 4);
    bufloc += 4;
    // The initial frame
    let mut coder = Coder::default();
    coder.unpack_array(
        &data[bufloc..],
        &mut quant,
        natoms * 3,
        initial_coding,
        initial_coding_parameter,
        natoms as usize,
    )?;

    // Skip past the actual data block.
    bufloc += u32::from(length) as usize;
    // Obtain the actual positions for the initial block.
    match initial_coding {
        TNG_COMPRESS_ALGO_POS_XTC2
        | TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE
        | TNG_COMPRESS_ALGO_POS_XTC3 => {
            if let Some(posdf) = posdf.as_deref_mut() {
                unquantize(
                    posdf,
                    natoms as usize,
                    1,
                    T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                    &quant,
                );
            }
            if let Some(posi) = posi.as_deref_mut() {
                posi[..natoms as usize * 3].copy_from_slice(&quant[..natoms as usize * 3]);
            }
        }
        TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA | TNG_COMPRESS_ALGO_POS_BWLZH_INTRA => {
            if let Some(posdf) = posdf.as_deref_mut() {
                unquantize_intra_differences(
                    posdf,
                    natoms as usize,
                    1,
                    T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                    &quant,
                );
            }
            if let Some(posi) = posi.as_deref_mut() {
                unquantize_intra_differences_int(posi, natoms as usize, 1, &quant);
            }

            unquantize_intra_differences_first_frame(&mut quant, natoms as usize);
        }
        _ => {}
    }
    // The remaining frames
    if nframes > 1 {
        bufloc += 4;
        coder = Coder::default();
        coder.unpack_array(
            &data[bufloc..],
            &mut quant[natoms as usize * 3..],
            (nframes - 1) * natoms * 3,
            coding,
            coding_parameter,
            natoms as usize,
        )?;

        match coding {
            TNG_COMPRESS_ALGO_POS_STOPBIT_INTER
            | TNG_COMPRESS_ALGO_POS_TRIPLET_INTER
            | TNG_COMPRESS_ALGO_POS_BWLZH_INTER => {
                // This requires that the first frame is already in one-to-one format, even if intra-frame
                // compression was done there. Therefore the unquant_intra_differences_first_frame should be called
                // before to convert it correctly.
                let natoms = natoms as usize;
                let nframes = nframes as usize;
                if let Some(posdf) = posdf {
                    unquantize_inter_differences(
                        posdf,
                        natoms,
                        nframes,
                        T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                        &quant,
                    );
                }
                if let Some(posi) = posi {
                    unquantize_inter_differences_int(posi, natoms, nframes, &quant);
                }
            }
            TNG_COMPRESS_ALGO_POS_XTC2
            | TNG_COMPRESS_ALGO_POS_XTC3
            | TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE => {
                let natoms = natoms as usize;
                let nframes = nframes as usize;
                if let Some(posdf) = posdf {
                    unquantize(
                        posdf,
                        natoms,
                        nframes,
                        T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                        &quant,
                    );
                }
                if let Some(posi) = posi {
                    posi[natoms * 3..].copy_from_slice(
                        &quant[natoms as usize * 3..natoms as usize * 3 * nframes],
                    );
                }
            }
            TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA | TNG_COMPRESS_ALGO_POS_BWLZH_INTRA => {
                let natoms = natoms as usize;
                let nframes = nframes as usize;
                if let Some(posdf) = posdf {
                    unquantize_intra_differences(
                        &mut posdf[natoms as usize * 3..],
                        natoms,
                        nframes - 1,
                        T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                        &quant[natoms as usize * 3..],
                    );
                }
                if let Some(posi) = posi {
                    unquantize_intra_differences_int(
                        &mut posi[natoms as usize * 3..],
                        natoms,
                        nframes - 1,
                        &quant[natoms as usize * 3..],
                    );
                }
            }
            _ => {}
        }
    }

    Ok((prec_hi, prec_lo))
}

fn compress_uncompress_vel<T: Float>(data: &[u8], vel: &mut [T]) -> Result<(), TngError> {
    let (_prec_hi, _prec_lo) = compress_uncompress_vel_gen(data, Some(vel), None)?;
    Ok(())
}

fn compress_uncompress_vel_int(data: &[u8], vel: &mut [i32]) -> Result<(FixT, FixT), TngError> {
    compress_uncompress_vel_gen::<f64>(data, None, Some(vel))
}

fn compress_uncompress_vel_gen<T: Float>(
    data: &[u8],
    mut veldf: Option<&mut [T]>,
    mut veli: Option<&mut [i32]>,
) -> Result<(FixT, FixT), TngError> {
    let mut bufloc = 0;

    // Magic integer for velocities
    let magic_int = u32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    if magic_int != MAGIC_INT_VEL {
        return Err(TngError::Constraint(format!(
            "Expected MAGIC_INT_VEL, got {magic_int}"
        )));
    }

    // Number of atoms
    let natoms = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Number of frames
    let nframes = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Initial coding
    let initial_coding = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Initial coding parameter
    let initial_coding_parameter = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Coding
    let coding = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Coding parameter.
    let coding_parameter = i32::from(readbufferfix(&data[bufloc..], 4));
    bufloc += 4;

    // Precision
    let prec_lo = readbufferfix(&data[bufloc..], 4);
    bufloc += 4;
    let prec_hi = readbufferfix(&data[bufloc..], 4);
    bufloc += 4;

    // Allocate the memory for the quantized positions
    let mut quant = vec![0; (natoms * nframes) as usize * 3];
    // The data block length
    let length = readbufferfix(&data[bufloc..], 4);
    bufloc += 4;
    // The initial frame
    let mut coder = Coder::default();
    coder.unpack_array(
        &data[bufloc..],
        &mut quant,
        natoms * 3,
        initial_coding,
        initial_coding_parameter,
        natoms as usize,
    )?;

    // Skip past the actual data block.
    bufloc += u32::from(length) as usize;
    // Obtain the actual positions for the initial block.
    match initial_coding {
        TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE
        | TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE
        | TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE => {
            if let Some(veldf) = veldf.as_deref_mut() {
                unquantize(
                    veldf,
                    natoms as usize,
                    1,
                    T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                    &quant,
                );
            }
            if let Some(veli) = veli.as_deref_mut() {
                veli[..natoms as usize * 3].copy_from_slice(&quant[..natoms as usize * 3]);
            }
        }
        _ => {}
    }
    // The remaining frames
    if nframes > 1 {
        bufloc += 4;
        coder = Coder::default();
        coder.unpack_array(
            &data[bufloc..],
            &mut quant[natoms as usize * 3..],
            (nframes - 1) * natoms * 3,
            coding,
            coding_parameter,
            natoms as usize,
        )?;

        // Inter-frame compression?
        match coding {
            TNG_COMPRESS_ALGO_VEL_TRIPLET_INTER
            | TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER
            | TNG_COMPRESS_ALGO_VEL_BWLZH_INTER => {
                // This requires that the first frame is already in one-to-one format.
                let natoms = natoms as usize;
                let nframes = nframes as usize;
                if let Some(veldf) = veldf {
                    unquantize_inter_differences(
                        veldf,
                        natoms,
                        nframes,
                        T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                        &quant,
                    );
                }
                if let Some(veli) = veli {
                    unquantize_inter_differences_int(veli, natoms, nframes, &quant);
                }
            }
            // One-to-one compression?
            TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE
            | TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE
            | TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE => {
                let natoms = natoms as usize;
                let nframes = nframes as usize;
                if let Some(veldf) = veldf {
                    unquantize(
                        veldf,
                        natoms,
                        nframes,
                        T::from_f64(fixt_pair_to_f64(prec_hi, prec_lo)),
                        &quant,
                    );
                }
                if let Some(veli) = veli {
                    veli[natoms * 3..].copy_from_slice(
                        &quant[natoms as usize * 3..natoms as usize * 3 * nframes],
                    );
                }
            }
            _ => {}
        }
    }

    Ok((prec_hi, prec_lo))
}
