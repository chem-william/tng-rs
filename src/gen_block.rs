use crate::{API_VERSION, MAX_STR_LEN, MD5_HASH_LEN};

/// Standard non-trajectory blocks
/// Block IDs of standard non-trajectory blocks
#[derive(Default, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[repr(u64)]
pub(crate) enum BlockID {
    // === Standard non-trajectory blocks ===
    GeneralInfo = 0x0000_0000_0000_0000,
    Molecules = 0x0000_0000_0000_0001,
    TrajectoryFrameSet = 0x0000_0000_0000_0002,
    ParticleMapping = 0x0000_0000_0000_0003,

    // === Standard trajectory blocks ===
    TrajBoxShape = 0x0000_0000_1000_0000,
    TrajPositions = 0x0000_0000_1000_0001,
    TrajVelocities = 0x0000_0000_1000_0002,
    TrajForces = 0x0000_0000_1000_0003,
    TrajPartialCharges = 0x0000_0000_1000_0004,
    TrajFormalCharges = 0x0000_0000_1000_0005,
    TrajBFactors = 0x0000_0000_1000_0006,
    TrajAnisotropicBFactors = 0x0000_0000_1000_0007,
    TrajOccupancy = 0x0000_0000_1000_0008,
    TrajGeneralComments = 0x0000_0000_1000_0009,
    TrajMasses = 0x0000_0000_1000_0010,

    // === GROMACS-specific blocks ===
    GmxLambda = 0x1000_0000_1000_0000,
    GmxEnergyAngle = 0x1000_0000_1000_0001,
    GmxEnergyRyckaertBell = 0x1000_0000_1000_0002,
    GmxEnergyLj14 = 0x1000_0000_1000_0003,
    GmxEnergyCoulomb14 = 0x1000_0000_1000_0004,
    GmxEnergyLjSr = 0x1000_0000_1000_0005,
    GmxEnergyCoulombSr = 0x1000_0000_1000_0006,
    GmxEnergyCoulRecip = 0x1000_0000_1000_0007,
    GmxEnergyPotential = 0x1000_0000_1000_0008,
    GmxEnergyKineticEn = 0x1000_0000_1000_0009,
    GmxEnergyTotalEnergy = 0x1000_0000_1000_0010,
    GmxEnergyTemperature = 0x1000_0000_1000_0011,
    GmxEnergyPressure = 0x1000_0000_1000_0012,
    GmxEnergyConstrRmsd = 0x1000_0000_1000_0013,
    GmxEnergyConstr2Rmsd = 0x1000_0000_1000_0014,
    GmxEnergyBoxX = 0x1000_0000_1000_0015,
    GmxEnergyBoxY = 0x1000_0000_1000_0016,
    GmxEnergyBoxZ = 0x1000_0000_1000_0017,
    GmxEnergyBoxxx = 0x1000_0000_1000_0018,
    GmxEnergyBoxyy = 0x1000_0000_1000_0019,
    GmxEnergyBoxzz = 0x1000_0000_1000_0020,
    GmxEnergyBoxyx = 0x1000_0000_1000_0021,
    GmxEnergyBoxzx = 0x1000_0000_1000_0022,
    GmxEnergyBoxzy = 0x1000_0000_1000_0023,
    GmxEnergyBoxvelxx = 0x1000_0000_1000_0024,
    GmxEnergyBoxvelyy = 0x1000_0000_1000_0025,
    GmxEnergyBoxvelzz = 0x1000_0000_1000_0026,
    GmxEnergyBoxvelyx = 0x1000_0000_1000_0027,
    GmxEnergyBoxvelzx = 0x1000_0000_1000_0028,
    GmxEnergyBoxvelzy = 0x1000_0000_1000_0029,
    GmxEnergyVolume = 0x1000_0000_1000_0030,
    GmxEnergyDensity = 0x1000_0000_1000_0031,
    GmxEnergyPv = 0x1000_0000_1000_0032,
    GmxEnergyEnthalpy = 0x1000_0000_1000_0033,
    GmxEnergyVirXx = 0x1000_0000_1000_0034,
    GmxEnergyVirXy = 0x1000_0000_1000_0035,
    GmxEnergyVirXz = 0x1000_0000_1000_0036,
    GmxEnergyVirYx = 0x1000_0000_1000_0037,
    GmxEnergyVirYy = 0x1000_0000_1000_0038,
    GmxEnergyVirYz = 0x1000_0000_1000_0039,
    GmxEnergyVirZx = 0x1000_0000_1000_0040,
    GmxEnergyVirZy = 0x1000_0000_1000_0041,
    GmxEnergyVirZz = 0x1000_0000_1000_0042,
    GmxEnergyShakevirXx = 0x1000_0000_1000_0043,
    GmxEnergyShakevirXy = 0x1000_0000_1000_0044,
    GmxEnergyShakevirXz = 0x1000_0000_1000_0045,
    GmxEnergyShakevirYx = 0x1000_0000_1000_0046,
    GmxEnergyShakevirYy = 0x1000_0000_1000_0047,
    GmxEnergyShakevirYz = 0x1000_0000_1000_0048,
    GmxEnergyShakevirZx = 0x1000_0000_1000_0049,
    GmxEnergyShakevirZy = 0x1000_0000_1000_0050,
    GmxEnergyShakevirZz = 0x1000_0000_1000_0051,
    GmxEnergyForcevirXx = 0x1000_0000_1000_0052,
    GmxEnergyForcevirXy = 0x1000_0000_1000_0053,
    GmxEnergyForcevirXz = 0x1000_0000_1000_0054,
    GmxEnergyForcevirYx = 0x1000_0000_1000_0055,
    GmxEnergyForcevirYy = 0x1000_0000_1000_0056,
    GmxEnergyForcevirYz = 0x1000_0000_1000_0057,
    GmxEnergyForcevirZx = 0x1000_0000_1000_0058,
    GmxEnergyForcevirZy = 0x1000_0000_1000_0059,
    GmxEnergyForcevirZz = 0x1000_0000_1000_0060,
    GmxEnergyPresXx = 0x1000_0000_1000_0061,
    GmxEnergyPresXy = 0x1000_0000_1000_0062,
    GmxEnergyPresXz = 0x1000_0000_1000_0063,
    GmxEnergyPresYx = 0x1000_0000_1000_0064,
    GmxEnergyPresYy = 0x1000_0000_1000_0065,
    GmxEnergyPresYz = 0x1000_0000_1000_0066,
    GmxEnergyPresZx = 0x1000_0000_1000_0067,
    GmxEnergyPresZy = 0x1000_0000_1000_0068,
    GmxEnergyPresZz = 0x1000_0000_1000_0069,
    GmxEnergySurfxsurften = 0x1000_0000_1000_0070,
    GmxEnergyMux = 0x1000_0000_1000_0071,
    GmxEnergyMuy = 0x1000_0000_1000_0072,
    GmxEnergyMuz = 0x1000_0000_1000_0073,
    GmxEnergyVcos = 0x1000_0000_1000_0074,
    GmxEnergyVisc = 0x1000_0000_1000_0075,
    GmxEnergyBarostat = 0x1000_0000_1000_0076,
    GmxEnergyTSystem = 0x1000_0000_1000_0077,
    GmxEnergyLambSystem = 0x1000_0000_1000_0078,
    GmxSelectionGroupNames = 0x1000_0000_1000_0079,
    GmxAtomSelectionGroup = 0x1000_0000_1000_0080,

    // === Fallback ===
    #[default]
    Unknown,
}

impl BlockID {
    /// Try to convert a raw i64 (e.g. read from file) into one of our variants.
    pub fn from_u64(raw: u64) -> Self {
        match raw {
            0x0000_0000_0000_0000 => BlockID::GeneralInfo,
            0x0000_0000_0000_0001 => BlockID::Molecules,
            0x0000_0000_0000_0002 => BlockID::TrajectoryFrameSet,
            0x0000_0000_0000_0003 => BlockID::ParticleMapping,

            // Standard trajectory blocks
            0x0000_0000_1000_0000 => Self::TrajBoxShape,
            0x0000_0000_1000_0001 => Self::TrajPositions,
            0x0000_0000_1000_0002 => Self::TrajVelocities,
            0x0000_0000_1000_0003 => Self::TrajForces,
            0x0000_0000_1000_0004 => Self::TrajPartialCharges,
            0x0000_0000_1000_0005 => Self::TrajFormalCharges,
            0x0000_0000_1000_0006 => Self::TrajBFactors,
            0x0000_0000_1000_0007 => Self::TrajAnisotropicBFactors,
            0x0000_0000_1000_0008 => Self::TrajOccupancy,
            0x0000_0000_1000_0009 => Self::TrajGeneralComments,
            0x0000_0000_1000_0010 => Self::TrajMasses,

            // GROMACS-specific blocks
            0x1000_0000_1000_0000 => Self::GmxLambda,
            0x1000_0000_1000_0001 => Self::GmxEnergyAngle,
            0x1000_0000_1000_0002 => Self::GmxEnergyRyckaertBell,
            0x1000_0000_1000_0003 => Self::GmxEnergyLj14,
            0x1000_0000_1000_0004 => Self::GmxEnergyCoulomb14,
            0x1000_0000_1000_0005 => Self::GmxEnergyLjSr,
            0x1000_0000_1000_0006 => Self::GmxEnergyCoulombSr,
            0x1000_0000_1000_0007 => Self::GmxEnergyCoulRecip,
            0x1000_0000_1000_0008 => Self::GmxEnergyPotential,
            0x1000_0000_1000_0009 => Self::GmxEnergyKineticEn,
            0x1000_0000_1000_0010 => Self::GmxEnergyTotalEnergy,
            0x1000_0000_1000_0011 => Self::GmxEnergyTemperature,
            0x1000_0000_1000_0012 => Self::GmxEnergyPressure,
            0x1000_0000_1000_0013 => Self::GmxEnergyConstrRmsd,
            0x1000_0000_1000_0014 => Self::GmxEnergyConstr2Rmsd,
            0x1000_0000_1000_0015 => Self::GmxEnergyBoxX,
            0x1000_0000_1000_0016 => Self::GmxEnergyBoxY,
            0x1000_0000_1000_0017 => Self::GmxEnergyBoxZ,
            0x1000_0000_1000_0018 => Self::GmxEnergyBoxxx,
            0x1000_0000_1000_0019 => Self::GmxEnergyBoxyy,
            0x1000_0000_1000_0020 => Self::GmxEnergyBoxzz,
            0x1000_0000_1000_0021 => Self::GmxEnergyBoxyx,
            0x1000_0000_1000_0022 => Self::GmxEnergyBoxzx,
            0x1000_0000_1000_0023 => Self::GmxEnergyBoxzy,
            0x1000_0000_1000_0024 => Self::GmxEnergyBoxvelxx,
            0x1000_0000_1000_0025 => Self::GmxEnergyBoxvelyy,
            0x1000_0000_1000_0026 => Self::GmxEnergyBoxvelzz,
            0x1000_0000_1000_0027 => Self::GmxEnergyBoxvelyx,
            0x1000_0000_1000_0028 => Self::GmxEnergyBoxvelzx,
            0x1000_0000_1000_0029 => Self::GmxEnergyBoxvelzy,
            0x1000_0000_1000_0030 => Self::GmxEnergyVolume,
            0x1000_0000_1000_0031 => Self::GmxEnergyDensity,
            0x1000_0000_1000_0032 => Self::GmxEnergyPv,
            0x1000_0000_1000_0033 => Self::GmxEnergyEnthalpy,
            0x1000_0000_1000_0034 => Self::GmxEnergyVirXx,
            0x1000_0000_1000_0035 => Self::GmxEnergyVirXy,
            0x1000_0000_1000_0036 => Self::GmxEnergyVirXz,
            0x1000_0000_1000_0037 => Self::GmxEnergyVirYx,
            0x1000_0000_1000_0038 => Self::GmxEnergyVirYy,
            0x1000_0000_1000_0039 => Self::GmxEnergyVirYz,
            0x1000_0000_1000_0040 => Self::GmxEnergyVirZx,
            0x1000_0000_1000_0041 => Self::GmxEnergyVirZy,
            0x1000_0000_1000_0042 => Self::GmxEnergyVirZz,
            0x1000_0000_1000_0043 => Self::GmxEnergyShakevirXx,
            0x1000_0000_1000_0044 => Self::GmxEnergyShakevirXy,
            0x1000_0000_1000_0045 => Self::GmxEnergyShakevirXz,
            0x1000_0000_1000_0046 => Self::GmxEnergyShakevirYx,
            0x1000_0000_1000_0047 => Self::GmxEnergyShakevirYy,
            0x1000_0000_1000_0048 => Self::GmxEnergyShakevirYz,
            0x1000_0000_1000_0049 => Self::GmxEnergyShakevirZx,
            0x1000_0000_1000_0050 => Self::GmxEnergyShakevirZy,
            0x1000_0000_1000_0051 => Self::GmxEnergyShakevirZz,
            0x1000_0000_1000_0052 => Self::GmxEnergyForcevirXx,
            0x1000_0000_1000_0053 => Self::GmxEnergyForcevirXy,
            0x1000_0000_1000_0054 => Self::GmxEnergyForcevirXz,
            0x1000_0000_1000_0055 => Self::GmxEnergyForcevirYx,
            0x1000_0000_1000_0056 => Self::GmxEnergyForcevirYy,
            0x1000_0000_1000_0057 => Self::GmxEnergyForcevirYz,
            0x1000_0000_1000_0058 => Self::GmxEnergyForcevirZx,
            0x1000_0000_1000_0059 => Self::GmxEnergyForcevirZy,
            0x1000_0000_1000_0060 => Self::GmxEnergyForcevirZz,
            0x1000_0000_1000_0061 => Self::GmxEnergyPresXx,
            0x1000_0000_1000_0062 => Self::GmxEnergyPresXy,
            0x1000_0000_1000_0063 => Self::GmxEnergyPresXz,
            0x1000_0000_1000_0064 => Self::GmxEnergyPresYx,
            0x1000_0000_1000_0065 => Self::GmxEnergyPresYy,
            0x1000_0000_1000_0066 => Self::GmxEnergyPresYz,
            0x1000_0000_1000_0067 => Self::GmxEnergyPresZx,
            0x1000_0000_1000_0068 => Self::GmxEnergyPresZy,
            0x1000_0000_1000_0069 => Self::GmxEnergyPresZz,
            0x1000_0000_1000_0070 => Self::GmxEnergySurfxsurften,
            0x1000_0000_1000_0071 => Self::GmxEnergyMux,
            0x1000_0000_1000_0072 => Self::GmxEnergyMuy,
            0x1000_0000_1000_0073 => Self::GmxEnergyMuz,
            0x1000_0000_1000_0074 => Self::GmxEnergyVcos,
            0x1000_0000_1000_0075 => Self::GmxEnergyVisc,
            0x1000_0000_1000_0076 => Self::GmxEnergyBarostat,
            0x1000_0000_1000_0077 => Self::GmxEnergyTSystem,
            0x1000_0000_1000_0078 => Self::GmxEnergyLambSystem,
            0x1000_0000_1000_0079 => Self::GmxSelectionGroupNames,
            0x1000_0000_1000_0080 => Self::GmxAtomSelectionGroup,
            _ => panic!("unknown block ID"),
        }
    }
}

#[derive(Debug, Default)]
pub struct GenBlock {
    /// The size of the block header in bytes
    pub header_contents_size: u64,

    /// The size of the block contents in bytes
    pub block_contents_size: u64,

    /// The ID of the block to determine its type
    pub(crate) id: BlockID,

    /// The MD5 hash of the block to verify integrity
    /// (fixed‐length array of exactly MD5_HASH_LEN bytes)
    pub md5_hash: [u8; MD5_HASH_LEN],

    /// The name of the block
    /// We wrap it as `Option<String>` so that `None` corresponds to a NULL pointer.
    pub name: Option<String>,

    /// The library version used to write the block
    pub version: u64,
    pub alt_hash_type: i64,
    pub alt_hash_len: i64,
    /// Alternative hash bytes (could be binary; `None` means NULL in C)
    pub alt_hash: Option<Vec<u8>>,

    /// Signature metadata
    pub signature_type: i64,
    pub signature_len: i64,
    /// Signature bytes (could be binary; `None` means NULL in C)
    pub signature: Option<Vec<u8>>,

    /// The full block header contents (arbitrary bytes; `None` ⇔ NULL pointer)
    pub header_contents: Option<Vec<u8>>,

    /// The full block contents (arbitrary bytes; `None` ⇔ NULL pointer)
    pub block_contents: Option<Vec<u8>>,
}

impl GenBlock {
    // c function name: tng_block_init
    pub fn new() -> Self {
        Self {
            id: BlockID::GeneralInfo,
            version: API_VERSION,
            ..Default::default()
        }
    }

    // c function: block_header_len_calculate
    /// Calculates the size (in bytes) of the block header, including
    /// fixed fields and the bounded name string length.
    ///
    /// If the block name is `None`, it will be treated as an empty string.
    ///
    /// Returns the total header size in bytes.
    pub(crate) fn calculate_header_len(&self) -> u64 {
        // Ensure we have at least an empty string
        let name = self.name.as_deref().unwrap_or("");
        let name_len = std::cmp::min(name.len() + 1, MAX_STR_LEN); // +1 for null-terminator

        let length = std::mem::size_of::<u64>()  // header_contents_size
        + std::mem::size_of::<u64>()  // block_contents_size
        + std::mem::size_of::<BlockID>() // id
        + std::mem::size_of::<u64>()   // block_version
        + MD5_HASH_LEN             // checksum
        + name_len; // name (null-terminated and bounded)
        u64::try_from(length).expect("u64 from usize")
    }
}
