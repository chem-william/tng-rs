use crate::{API_VERSION, MD5_HASH_LEN};

/// Standard non-trajectory blocks
/// Block IDs of standard non-trajectory blocks
#[repr(i64)]
#[derive(Debug, Default)]
pub(crate) enum BlockID {
    #[default]
    Undetermined = 0x1111_1111_1111_1111,
    // 0x0000000000000000LL
    GeneralInfo = 0x0000_0000_0000_0000,
    // 0x0000000000000001LL
    Molecules = 0x0000_0000_0000_0001,
    // 0x0000000000000002LL
    TrajectoryFrameSet = 0x0000_0000_0000_0002,
    // 0x0000000000000003LL
    ParticleMapping = 0x0000_0000_0000_0003,
}

impl BlockID {
    /// Try to convert a raw i64 (e.g. read from file) into one of our variants.
    pub fn from_i64(raw: i64) -> Self {
        match raw {
            0x0000_0000_0000_0000 => BlockID::GeneralInfo,
            0x0000_0000_0000_0001 => BlockID::Molecules,
            0x0000_0000_0000_0002 => BlockID::TrajectoryFrameSet,
            0x0000_0000_0000_0003 => BlockID::ParticleMapping,
            _ => panic!("unknown block ID"),
        }
    }
}

#[derive(Debug, Default)]
pub struct GenBlock {
    /// The size of the block header in bytes
    pub header_contents_size: i64,

    /// The size of the block contents in bytes
    pub block_contents_size: i64,

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
}
