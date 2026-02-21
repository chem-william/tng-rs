use std::cmp::{Ordering, Reverse};

use log::debug;

const MAX_HUFFMAN_LEN: usize = 31;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bit {
    #[default]
    Zero = 0,
    One = 1,
}

#[derive(Debug, Default)]
pub struct InvalidBit(u8);

impl TryFrom<u8> for Bit {
    type Error = InvalidBit;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Bit::Zero),
            1 => Ok(Bit::One),
            _ => panic!("invalid bit"),
        }
    }
}

/// A leaf node
#[derive(Debug, Default, Clone)]
pub struct HTreeLeaf {
    /// Index into input dictionary
    pub idict: usize,
    pub prob: u32,
    pub bit: Bit,
}

/// An internal node
#[derive(Debug, Clone)]
pub struct HTreeNode {
    pub left: Box<HTreeNodeLeaf>,
    pub right: Box<HTreeNodeLeaf>,
    pub bit: Bit,
    pub prob: u32,
}

#[derive(Debug, Clone)]
pub(crate) enum HTreeNodeLeaf {
    Leaf(HTreeLeaf),
    Node(HTreeNode),
}

impl Default for HTreeNodeLeaf {
    fn default() -> Self {
        Self::Leaf(HTreeLeaf::default())
    }
}

impl HTreeNodeLeaf {
    pub fn prob(&self) -> u32 {
        match self {
            HTreeNodeLeaf::Leaf(leaf) => leaf.prob,
            HTreeNodeLeaf::Node(node) => node.prob,
        }
    }

    fn set_bit(&mut self, bit: Bit) {
        match self {
            HTreeNodeLeaf::Leaf(leaf) => leaf.bit = bit,
            HTreeNodeLeaf::Node(node) => node.bit = bit,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CodeLength {
    pub code: u32,
    pub length: usize,
    pub dict: u32,
    pub prob: u32,
}

/// The `huffman_dict` array should be 131077 (0x20005) long. The `huffman_dict_unpacked` array
/// should be 131077 long (note five longer than 0x20000)
pub(crate) fn ptngc_comp_conv_to_huffman(
    vals: &[u32],
    dict: &[u32],
    prob: &mut [u32],
    huffman: &mut [u8],
    huffman_len: &mut usize,
    huffman_dict: &mut [u8],
    huffman_dictlen: &mut usize,
    huffman_dict_unpacked: &mut [u32],
    huffman_dict_unpackedlen: &mut usize,
) {
    let ndict = dict.len();
    let mut bitptr = 0;
    let mut huffman_index = 0usize; // instead of huffman_ptr
    let mut longcodes = true;
    let mut codelength: Vec<CodeLength> = vec![CodeLength::default(); ndict];

    while longcodes {
        // Create array of leafs (will be array of nodes/trees during buildup of tree)
        let mut htree = Vec::with_capacity(ndict);
        bitptr = 0;
        // Instead of copying the address of `huffman`, we just the index to 0.
        // in the original code, `huffman` and `huffman_ptr` would point to the same
        // location in memory (at the start of the array)
        huffman_index = 0usize;

        for (i, &p) in prob[..ndict].iter().enumerate() {
            htree.push(HTreeNodeLeaf::Leaf(HTreeLeaf {
                idict: i,
                prob: p,
                bit: Bit::Zero,
            }));
        }

        // Sort the leafs wrt probability
        htree.sort_by_key(|item| Reverse(item.prob()));

        // Build the tree
        if ndict == 1 {
            codelength[0].code = 1;
            codelength[0].length = 1;
        } else {
            // Nodes and leafs left
            let mut nleft = ndict;

            // Take the two least probable symbols (which are at the end of the array and combine
            // them until there is nothing left)
            while nleft > 1 {
                let mut n1 = Box::new(htree[nleft - 1].clone());
                let mut n2 = Box::new(htree[nleft - 2].clone());

                // assign bits
                n1.set_bit(Bit::Zero);
                n2.set_bit(Bit::One);

                let new_prob = n1.prob() + n2.prob();
                nleft -= 1;

                // Create a new node
                htree[nleft - 1] = HTreeNodeLeaf::Node(HTreeNode {
                    left: n1,
                    right: n2,
                    bit: Bit::Zero,
                    prob: new_prob,
                });

                let mut new_place = nleft;

                while new_place > 0 {
                    let pc = htree[new_place - 1].prob();
                    if new_prob < pc {
                        break;
                    } else {
                        new_place -= 1;
                    }
                }

                if new_place != nleft {
                    htree[new_place..nleft].rotate_right(1);
                }
            }
        }

        // Create codes from tree
        if let Some(first) = htree.first() {
            assign_codes(first, codelength.as_mut_slice(), 0, 0, true);
        } else {
            return;
        }

        // Canonicalize
        // First put values into to `codelength` for sorting
        codelength
            .iter_mut()
            .zip(dict)
            .zip(&mut *prob)
            .for_each(|((code, d), p)| {
                code.dict = *d;
                code.prob = *p;
            });

        // Sort codes wrt length/value
        // codelength.sort_by_key(|c| (c.length, c.dict));
        codelength.sort_by(|a, b| {
            if a.length > b.length {
                Ordering::Greater
            } else if a.length < b.length {
                Ordering::Less
            } else if a.dict > b.dict {
                Ordering::Greater
            } else {
                // covers both a.dict < b.dict AND a.dict == b.dict
                Ordering::Less
            }
        });

        // Canonicalize codes
        let mut code = 0;
        for i in 0..ndict {
            codelength[i].code = code;
            if i < ndict - 1 {
                code = (code + 1) << (codelength[i + 1].length - codelength[i].length);
            }
        }

        longcodes = codelength
            .iter()
            .take(ndict)
            .any(|c| c.length > MAX_HUFFMAN_LEN);

        // If the codes are too long alter the probabilities
        if longcodes {
            prob.iter_mut().for_each(|p| *p = (*p >> 1).max(1));
        }
    }

    // Simply do compression by writing out the bits
    for dict_val in vals {
        let r = codelength
            .iter()
            .position(|cl| cl.dict == *dict_val)
            .unwrap_or(ndict);
        writebits(
            codelength[r].code,
            codelength[r].length,
            huffman,
            &mut huffman_index,
            &mut bitptr,
        );
    }

    if bitptr != 0 {
        writebits(0, 8 - bitptr, huffman, &mut huffman_index, &mut bitptr);
    }

    *huffman_len = huffman_index;

    // Output dictionary
    //First the largest symbol value is written in 16 bits. No bits are
    //  encoded for symbols larger than this.  Then one bit signifies if
    //  there is a used symbol: 1 If unused entry: 0 If used symbol the 5
    //  following bits encode the length of the symbol. Worst case is
    //  thus 6*65538 bits used for the dictionary. That won't happen
    //  unless there's really that many values in use. If that is so,
    //  well, either we compress well, or we have many values anyway.
    // First sort the dictionary wrt symbol
    // codelength.sort_by_key(|c| c.dict);
    codelength.sort_by(|a, b| {
        if a.dict > b.dict {
            Ordering::Greater
        } else if a.dict < b.dict {
            Ordering::Less
        } else {
            // covers both a.dict < b.dict AND a.dict == b.dict
            Ordering::Less
        }
    });

    bitptr = 0;
    let mut huffman_dict_index = 0usize; // Instead of the huffman_ptr
    let dict_value = codelength[ndict - 1].dict;

    // Extract bytes from the 32-bit dict value
    let byte0 = (dict_value & 0xFFu32) as u8; // Low byte
    let byte1 = ((dict_value >> 8) & 0xFFu32) as u8; // Second byte  
    let byte2 = ((dict_value >> 16) & 0xFFu32) as u8; // Third byte

    // Write to huffman_dict (equivalent to the *huffman_ptr++ operations)
    huffman_dict[huffman_dict_index] = byte0;
    huffman_dict_index += 1;
    huffman_dict[huffman_dict_index] = byte1;
    huffman_dict_index += 1;
    huffman_dict[huffman_dict_index] = byte2;
    huffman_dict_index += 1;

    // Write to huffman_dict_unpacked
    huffman_dict_unpacked[0] = byte0 as u32;
    huffman_dict_unpacked[1] = byte1 as u32;
    huffman_dict_unpacked[2] = byte2 as u32;

    for i in 0..codelength[ndict - 1].dict + 1 {
        // Do I have this value?
        let mut ihave = false;
        for cl in codelength.iter().take(ndict) {
            if cl.dict == i {
                ihave = true;
                writebits(1, 1, huffman_dict, &mut huffman_dict_index, &mut bitptr);
                writebits(
                    u32::try_from(cl.length).expect("u32 from usize"),
                    5,
                    huffman_dict,
                    &mut huffman_dict_index,
                    &mut bitptr,
                );
                huffman_dict_unpacked[usize::try_from(3 + i).expect("usize from u32")] =
                    u32::try_from(cl.length).expect("u32 from usize");
                break;
            }
        }
        if !ihave {
            writebits(0, 1, huffman_dict, &mut huffman_dict_index, &mut bitptr);
            huffman_dict_unpacked[usize::try_from(3 + i).expect("usize from u32")] = 0;
        }
    }
    if bitptr != 0 {
        writebits(
            0,
            8 - bitptr,
            huffman_dict,
            &mut huffman_dict_index,
            &mut bitptr,
        );
    }

    *huffman_dictlen = huffman_dict_index;
    *huffman_dict_unpackedlen =
        usize::try_from(3 + codelength[ndict - 1].dict + 1).expect("usize from u32");
}

fn flush_8bits(combine: &mut u32, output: &mut [u8], output_index: &mut usize, bitptr: &mut usize) {
    while *bitptr >= 8 {
        let mask = !(0xFFu32 << (*bitptr - 8));
        let out = (*combine >> (*bitptr - 8)) as u8;
        if *output_index < output.len() {
            output[*output_index] = out;
            *output_index += 1;
        }
        *bitptr -= 8;
        *combine &= mask;
    }
}

fn writebits(
    value: u32,
    mut length: usize,
    output: &mut [u8],
    output_index: &mut usize,
    bitptr: &mut usize,
) {
    // Read current byte from output position
    let mut combine = output[*output_index] as u32;
    // let mut combine: u32 = if *output_index < output.len() {
    //     output[*output_index] as u32
    // } else {
    //     0
    // };

    let mut mask: u32;
    if length >= 8 {
        mask = 0xFFu32 << (length - 8);
    } else {
        mask = 0xFFu32 >> (8 - length);
    }

    while length > 8 {
        // Make room for the bits
        combine <<= 8;
        *bitptr += 8;
        combine |= (value & mask) >> (length - 8);
        flush_8bits(&mut combine, output, output_index, bitptr);
        length -= 8;
        mask >>= 8;
    }

    if length != 0 {
        // Make room for the bits
        combine <<= length;
        *bitptr += length;
        combine |= value;
        flush_8bits(&mut combine, output, output_index, bitptr);
    }

    // Write back to current position
    if *output_index < output.len() {
        output[*output_index] = combine as u8;
    }
}

fn assign_codes(
    htree: &HTreeNodeLeaf,
    codelength: &mut [CodeLength],
    mut code: u32,
    mut length: usize,
    top: bool,
) {
    debug!("Assign codes called with code {code} length {length}");
    match htree {
        HTreeNodeLeaf::Leaf(leaf) => {
            codelength[leaf.idict].length = length + 1;
            codelength[leaf.idict].code = (code << 1) | (leaf.bit as u32);

            debug!(
                "I am a leaf: {} {}",
                codelength[leaf.idict].length, codelength[leaf.idict].code
            );
        }
        HTreeNodeLeaf::Node(node) => {
            if !top {
                code <<= 1;
                code |= node.bit as u32;
                length += 1;
            }
            debug!("I am a node length: {length}");
            debug!("I am a node code: {code}");
            assign_codes(&node.left, codelength, code, length, false);
            assign_codes(&node.right, codelength, code, length, false);
        }
    }
}
