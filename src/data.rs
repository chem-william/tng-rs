use crate::gen_block::BlockID;
use std::cmp::max;

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

    pub fn get_size(self) -> usize {
        match self {
            DataType::Char => 1,
            DataType::Int => size_of::<i64>(),
            DataType::Float => size_of::<f32>(),
            DataType::Double => size_of::<f64>(),
        }
    }
}

/// Compression mode is specified in each data block
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Compression {
    #[default]
    Uncompressed = 0,
    XTC = 1,
    TNG = 2,
    GZip = 3,
}

impl Compression {
    /// Try to interpret a raw i64 as one of our variants. Unknown values become `Uncompressed`
    pub fn from_i64(raw: i64) -> Self {
        match raw {
            1 => Compression::XTC,
            2 => Compression::TNG,
            3 => Compression::GZip,
            _ => Compression::Uncompressed, // fallback
        }
    }
}

/// Indicates what kind of dependency a data block has.
/// In C, `dependency` was a `char` flag—here we model the common cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dependency {
    /// Data varies per frame.
    FrameDependent,
    /// Data varies per particle.
    ParticleDependent,
    /// Data varies per frame _and_ per particle.
    FrameAndParticleDependent,
    /// No frame‐ or particle‐dependency (static data).
    Independent,
}

#[derive(Debug, Clone, Default)]
pub struct Data {
    /// The block ID of this data block (identifies the type of data).
    pub block_id: BlockID,

    /// The name of the data block. This is used to determine the kind of data is stored.
    pub block_name: String,

    /// The type of data stored in `values`
    pub data_type: DataType,

    /// Frame/particle dependency flag.
    pub dependency: u8,

    /// The frame number at which data begins.
    pub first_frame_with_data: i64,
    /// The total number of frames in this data block.
    pub n_frames: i64,
    /// The number of values stored per frame.
    pub n_values_per_frame: i64,
    /// The number of frames between each data point - e.g. when storing sparse data.
    pub stride_length: i64,

    /// ID of the CODEC used for compression. 0 == no compression
    pub codec_id: Compression,
    /// If reading one frame at a time, this is the last frame read.
    pub last_retrieved_frame: i64,

    /// Multiplier used for getting integer values for compression
    pub compression_multiplier: f64,

    /// Numeric data values, if any. Each entry corresponds to one data point.
    /// The total length should be `n_frames * n_particles * n_values_per_frame`,
    /// where `n_particles` comes from the enclosing frame set.
    /// A 1-dimensional array of values of length
    ///  `n_frames` * `n_particles`* `n_values_per_frame`
    // pub values: Option<DataValue>,
    pub values: Option<Vec<u8>>,

    /// Character‐based data (e.g. labels). Modeled as a 3D array of `String`:
    pub strings: Option<Vec<Vec<Vec<String>>>>,
}

impl Data {
    /// Allocate memory for storing particle data. The allocated block will be referred to by data->values.
    ///
    /// # Panic
    /// Panics if we run out of memory when trying to allocate `size * frame_alloc * n_particles * n_values_per_frame`.
    pub fn allocate_particle_data_mem(
        &mut self,
        n_frames: i64,
        stride_length: i64,
        n_particles: i64,
        n_values_per_frame: i64,
    ) {
        assert!(
            n_particles != 0 && n_values_per_frame != 0,
            "n_particles == 0 || n_values_per_frame == 0"
        );

        if self.strings.is_some() && self.data_type == DataType::Char {
            self.strings = None;
        }

        self.n_frames = n_frames;
        let eff_n_frames = max(1, n_frames);
        self.stride_length = max(1, stride_length);
        self.n_values_per_frame = n_values_per_frame;
        let frame_alloc = ((eff_n_frames - 1) / stride_length + 1) as usize;

        if self.data_type == DataType::Char {
            // This will panic on OOM.
            let frames: Vec<Vec<Vec<String>>> =
                vec![
                    vec![vec![String::new(); n_values_per_frame as usize]; n_particles as usize];
                    frame_alloc
                ];
            self.strings = Some(frames);
        } else {
            let size = match self.data_type {
                DataType::Int => size_of::<i64>(),
                DataType::Float => size_of::<f32>(),
                DataType::Double => size_of::<f64>(),
                DataType::Char => unreachable!(),
            };

            // Compute total length: `size * frame_alloc * n_particles * n_values_per_frame`.
            let total_len = frame_alloc
                .checked_mul(usize::try_from(n_particles).expect("i64 to usize"))
                .and_then(|x| {
                    x.checked_mul(usize::try_from(n_values_per_frame).expect("i64 to usize"))
                })
                .and_then(|x| x.checked_mul(size))
                .unwrap_or(0);

            // One‐shot allocate zeroed Vec<u8>. Panics on OOM.
            let buf: Vec<u8> = vec![0u8; total_len];
            self.values = Some(buf);
        }
    }

    /// Allocate memory for storing non-particle data. The allocated block will be referred to by data->values.
    ///
    /// # Panic
    /// Panics if we run out of memory when trying to allocate `strings` of size `size * frame_alloc * n_values_per_frame`.
    pub fn allocate_data_mem(
        &mut self,
        n_frames: i64,
        stride_length: i64,
        n_values_per_frame: i64,
    ) {
        assert!(
            n_values_per_frame != 0,
            "n_particles == 0 || n_values_per_frame == 0"
        );

        if self.strings.is_some() && self.data_type == DataType::Char {
            self.strings = None;
        }

        self.n_frames = n_frames;
        let eff_n_frames = max(1, n_frames);
        self.stride_length = max(1, stride_length);
        self.n_values_per_frame = n_values_per_frame;
        let frame_alloc = ((eff_n_frames - 1) / self.stride_length + 1) as usize;

        if self.data_type == DataType::Char {
            // This will panic on OOM.
            let frames_for_group: Vec<Vec<String>> =
                vec![vec![String::new(); n_values_per_frame as usize]; frame_alloc];
            let all_groups: Vec<Vec<Vec<String>>> = vec![frames_for_group];
            self.strings = Some(all_groups);
        } else {
            let size = match self.data_type {
                DataType::Int => size_of::<i64>(),
                DataType::Float => size_of::<f32>(),
                DataType::Double => size_of::<f64>(),
                DataType::Char => size_of::<u8>(), // not used here
            };

            // Compute total length: `elem_size * frame_alloc * n_values_per_frame`.
            let total_len = frame_alloc
                .checked_mul(usize::try_from(n_values_per_frame).expect("i64 to usize"))
                .and_then(|x| x.checked_mul(size))
                .unwrap_or(0);

            // One‐shot allocate a zeroed Vec<u8>. Panics on OOM.
            let buf: Vec<u8> = vec![0u8; total_len];
            self.values = Some(buf);
        }
    }
}
