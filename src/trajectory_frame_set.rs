use crate::data::Data;
use crate::particle_mapping::ParticleMapping;

#[derive(Debug)]
pub struct TrajectoryFrameSet {
    /// The number of different particle mapping blocks present.
    pub n_mapping_blocks: i64,
    /// The atom mappings of this frame set.
    pub mappings: Vec<ParticleMapping>,

    /// The first frame of this frame set.
    pub first_frame: i64,
    /// The number of frames in this frame set.
    pub n_frames: i64,
    /// The number of written frames in this frame set (used when writing one frame at a time).
    pub n_written_frames: i64,
    /// The number of frames not yet written to file in this frame set
    /// (used from the utility functions to finish the writing properly).
    pub n_unwritten_frames: i64,

    /// A list of the count of each molecule type—only used when using variable number of atoms.
    pub molecule_cnt_list: Vec<i64>,
    /// The number of particles/atoms—only used when using variable number of atoms.
    pub n_particles: i64,

    /// The file position (in bytes) of the next frame set.
    pub next_frame_set_file_pos: i64,
    /// The file position (in bytes) of the previous frame set.
    pub prev_frame_set_file_pos: i64,
    /// The file position (in bytes) of the frame set one medium‐stride ahead.
    pub medium_stride_next_frame_set_file_pos: i64,
    /// The file position (in bytes) of the frame set one medium‐stride behind.
    pub medium_stride_prev_frame_set_file_pos: i64,
    /// The file position (in bytes) of the frame set one long‐stride ahead.
    pub long_stride_next_frame_set_file_pos: i64,
    /// The file position (in bytes) of the frame set one long‐stride behind.
    pub long_stride_prev_frame_set_file_pos: i64,

    /// Time stamp (in seconds) of the first frame in this frame set.
    pub first_frame_time: f64,

    /// The number of trajectory data blocks of particle‐dependent data.
    pub n_particle_data_blocks: usize,
    /// A list of data blocks containing particle‐dependent trajectory data.
    pub tr_particle_data: Vec<Data>,

    /// The number of trajectory data blocks independent of particles.
    pub n_data_blocks: usize,
    /// A list of data blocks containing frame‐ and particle‐independent trajectory data.
    pub tr_data: Vec<Data>,
}

impl TrajectoryFrameSet {
    pub fn new() -> Self {
        Self {
            n_mapping_blocks: 0,
            mappings: Vec::new(),

            first_frame: -1,
            n_frames: 0,
            n_written_frames: 0,
            n_unwritten_frames: 0,

            molecule_cnt_list: Vec::new(),
            n_particles: 0,

            next_frame_set_file_pos: -1,
            prev_frame_set_file_pos: -1,
            medium_stride_next_frame_set_file_pos: -1,
            medium_stride_prev_frame_set_file_pos: -1,
            long_stride_next_frame_set_file_pos: -1,
            long_stride_prev_frame_set_file_pos: -1,

            first_frame_time: -1.0,

            n_particle_data_blocks: 0,
            tr_particle_data: Vec::new(),

            n_data_blocks: 0,
            tr_data: Vec::new(),
        }
    }
}
