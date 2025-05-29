/// Describes a block of particle indexes and their “real” particle IDs.
#[derive(Debug, Clone)]
pub struct ParticleMapping {
    /// The index number of the first particle in this mapping block
    pub num_first_particle: i64,

    /// The number of particles list in this mapping block
    pub n_particles: i64,

    /// the mapping of index numbers to the real particle numbers in the
    /// trajectory. real_particle_numbers[0] is the real particle number
    /// (as it is numbered in the molecular system) of the first particle
    /// in the data blocks covered by this particle mapping block
    pub real_particle_numbers: Vec<i64>,
}

impl ParticleMapping {
    pub fn new() -> Self {
        Self {
            num_first_particle: 0,
            n_particles: 0,
            real_particle_numbers: Vec::new(),
        }
    }
}
