/// Possible formats of data block contents
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Char,
    Int,
    Float,
    Double,
}

#[derive(Debug)]
enum DataValue {
    Int(Vec<i64>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

/// Compression mode is specified in each data block
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    Uncompressed,
    XTCCompression,
    TNGCompression,
    GZipCompression,
}

/// Indicates what kind of dependency a data block has.
/// In C, `dependency` was a `char` flag—here we model the common cases.
/// Adjust variants as needed to match your library’s semantics.
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

impl Default for Dependency {
    fn default() -> Self {
        Dependency::Independent
    }
}

#[derive(Debug)]
pub struct Data {
    /// The block ID of this data block (identifies the type of data).
    pub block_id: i64,

    /// The name of the data block. This is used to determine the kind of data is stored.
    pub block_name: String,

    /// The type of data stored in `values`
    pub data_type: DataType,

    /// Frame/particle dependency flag.
    pub dependency: Dependency,

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
    /// where “n_particles” comes from the enclosing frame set.
    /// A 1-dimensional array of values of length
    ///  n_frames * n_particles * n_values_per_frame
    pub values: Option<DataValue>,

    /// Character‐based data (e.g. labels). Modeled as a 3D array of `String`:
    pub strings: Option<Vec<Vec<Vec<String>>>>,
}

impl Data {
    pub fn new(block_id: i64, block_name: impl Into<String>, data_type: DataType) -> Self {
        Data {
            block_id,
            block_name: block_name.into(),
            data_type,
            dependency: Dependency::default(),
            first_frame_with_data: 0,
            n_frames: 0,
            n_values_per_frame: 0,
            stride_length: 1,
            codec_id: Compression::Uncompressed,
            last_retrieved_frame: -1,
            compression_multiplier: 1.0,
            values: None,
            strings: None,
        }
    }

    pub fn allocate_values(
        &mut self,
        frame_alloc: usize,
        n_particles: usize,
        n_values_per_frame: usize,
    ) {
        // Compute total = frame_alloc * n_particles * n_values_per_frame,
        // checking for overflow just like C would (undefined on overflow).
        let total = frame_alloc
            .checked_mul(n_particles)
            .and_then(|t| t.checked_mul(n_values_per_frame))
            .expect("no overflow");

        match self.data_type {
            DataType::Int => {
                // Create a fresh Vec<i64>
                let mut vec: Vec<i64> = Vec::new();
                // Try to reserve `total` elements; if this fails, bail out:
                vec.try_reserve(total).expect("able to reserve for int");
                // Now actually set length (equivalent to zero‐inited or uninitialized C memory,
                // but here we fill with zeroes for safety; adjust as needed):
                vec.resize(total, 0_i64);
                self.values = Some(DataValue::Int(vec));
            }

            DataType::Float => {
                let mut vec: Vec<f32> = Vec::new();
                vec.try_reserve(total).expect("able to reserve for Float");
                vec.resize(total, 0.0_f32);
                self.values = Some(DataValue::Float(vec));
            }

            DataType::Double => {
                let mut vec: Vec<f64> = Vec::new();
                vec.try_reserve(total).expect("able to reserve for Double");
                vec.resize(total, 0.0_f64);
                self.values = Some(DataValue::Double(vec));
            }
            DataType::Char => todo!("haven't implemented Char type"),
        }
    }

    /// Initialize the 3D `strings` array with the given dimensions.
    /// Example usage:
    // / ```
    /// // frames x particles x values_per_frame
    /// data.strings = Some(vec![
    ///     vec![vec![String::new(); values_per_frame]; n_particles];
    ///     n_frames
    /// ]);
    // / ```
    pub fn init_strings(&mut self, n_frames: usize, n_particles: usize, values_per_frame: usize) {
        let mut outer = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            let mut per_frame = Vec::with_capacity(n_particles);
            for _ in 0..n_particles {
                // Initialize a Vec<String> of length `values_per_frame`
                per_frame.push(vec![String::new(); values_per_frame]);
            }
            outer.push(per_frame);
        }
        self.strings = Some(outer);
    }
}
