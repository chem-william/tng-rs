use crate::{
    bwlzh::{N_HUFFMAN_ALGO, ptngc_comp_conv_from_vals16, ptngc_comp_conv_to_vals16},
    dict::ptngc_comp_make_dict_hist,
    huffman::{
        HUFFMAN_DICT_CAP, HUFFMAN_DICT_UNPACKED_CAP, ptngc_comp_conv_from_huffman,
        ptngc_comp_conv_to_huffman,
    },
    rle::{ptngc_comp_conv_from_rle, ptngc_comp_conv_to_rle},
};

pub(crate) const fn ptngc_comp_huff_buflen(nvals: usize) -> usize {
    132000 + nvals * 8
}

const HUFF_ALGO_NAMES: [&str; 3] = [
    "Huffman (dict=raw)",
    "Huffman (dict=Huffman)",
    "Huffman (dict=RLE+Huffman)",
];
pub const fn ptngc_comp_get_huff_algo_name(algo: usize) -> Option<&'static str> {
    if algo >= N_HUFFMAN_ALGO {
        return None;
    };
    Some(HUFF_ALGO_NAMES[algo])
}

// The value pointed to be `chosen_algo` should be sent as -1 for autodetect
pub(crate) fn ptngc_comp_huff_compress_verbose(
    vals: &mut [u32],
    huffman: &mut [u8],
    huffman_len: &mut i32,
    huffdatalen: &mut usize,
    huffman_lengths: &mut [usize],
    chosen_algo: &mut i32,
    isvals16: bool,
) {
    let nvals16;
    let mut nvals = vals.len();

    // Do I need to convert to vals16?
    if !isvals16 {
        let mut vals16 = vec![0; nvals * 3];
        nvals16 = ptngc_comp_conv_to_vals16(vals, &mut vals16);
        nvals = nvals16;
        vals.clone_from_slice(&vals16[..nvals]);
    } else {
        nvals16 = nvals;
    }

    // Determine probabilities
    let (dict, mut hist) = ptngc_comp_make_dict_hist(vals);

    // First compress the data using huffman coding (place it ready for output at 14 (code for algorithm+length etc.))
    let mut nhuff = 0;
    let mut nhuffdict = 0;
    let mut nhuffdictunpack = 0;
    let mut huffdict = vec![0; HUFFMAN_DICT_CAP];
    let mut huffdictunpack = vec![0; HUFFMAN_DICT_UNPACKED_CAP];
    ptngc_comp_conv_to_huffman(
        vals,
        &dict,
        &mut hist,
        &mut huffman[14..],
        &mut nhuff,
        &mut huffdict,
        &mut nhuffdict,
        &mut huffdictunpack,
        &mut nhuffdictunpack,
    );
    let ndict = dict.len();
    *huffdatalen = nhuff;

    // Algorithm 0 stores the huffman dictionary directly (+ a code for
    // the algorithm) + lengths of the huffman buffer (4) and the huffman dictionary (3).
    huffman_lengths[0] = nhuff + nhuffdict + 2 + 3 * 4 + 3 + 3;

    // Next we try to compress the huffman dictionary using huffman coding ... (algorithm 1)

    // Determine probabilities
    let (dict, mut hist) = ptngc_comp_make_dict_hist(&huffdictunpack[..nhuffdictunpack]);
    let ndict1 = dict.len();
    // Pack huffman dictionary
    let mut nhuff1 = 0;
    let mut nhuffdict1 = 0;
    let mut nhuffdictunpack1 = 0;
    let mut huffman1 = vec![0; 2 * HUFFMAN_DICT_UNPACKED_CAP];
    let mut huffdict1 = vec![0; HUFFMAN_DICT_UNPACKED_CAP];
    let mut huffdictunpack1 = vec![0; HUFFMAN_DICT_UNPACKED_CAP];
    ptngc_comp_conv_to_huffman(
        &huffdictunpack[..nhuffdictunpack],
        &dict,
        &mut hist,
        &mut huffman1,
        &mut nhuff1,
        &mut huffdict1,
        &mut nhuffdict1,
        &mut huffdictunpack1,
        &mut nhuffdictunpack1,
    );
    huffman_lengths[1] = nhuff + nhuff1 + nhuffdict1 + 2 + 3 * 4 + 3 + 3 + 3 + 3 + 3;

    // ... and rle + huffman coding ... (algorithm 2) Pack any repetitive patterns
    let huffdictrle = ptngc_comp_conv_to_rle(&huffdictunpack[..nhuffdictunpack], 1);

    // Determine probabilities
    let (dict, mut hist) = ptngc_comp_make_dict_hist(&huffdictrle);
    let ndict2 = dict.len();
    // Pack huffman dictionary
    let mut huffman2 = vec![0; 6 * HUFFMAN_DICT_UNPACKED_CAP];
    let mut huffdict2 = vec![0; HUFFMAN_DICT_UNPACKED_CAP];
    let mut huffdictunpack2 = vec![0; HUFFMAN_DICT_UNPACKED_CAP];
    let mut nhuff2 = 0;
    let mut nhuffdict2 = 0;
    let mut nhuffdictunpack2 = 0;
    ptngc_comp_conv_to_huffman(
        &huffdictrle,
        &dict,
        &mut hist,
        &mut huffman2,
        &mut nhuff2,
        &mut huffdict2,
        &mut nhuffdict2,
        &mut huffdictunpack2,
        &mut nhuffdictunpack2,
    );
    let nhuffrle = huffdictrle.len();
    huffman_lengths[2] = nhuff + nhuff2 + nhuffdict2 + 2 + 3 * 4 + 3 + 3 + 3 + 3 + 3 + 3;

    // Choose the best algorithm and output the data
    if (*chosen_algo == 0)
        || ((*chosen_algo == -1)
            && ((huffman_lengths[0] < huffman_lengths[1])
                && (huffman_lengths[0] < huffman_lengths[2])))
    {
        *chosen_algo = 0;
        *huffman_len = i32::try_from(huffman_lengths[0]).expect("i32 from usize");
        huffman[0] = isvals16 as u8;
        huffman[1] = 0;
        huffman[2] = (nvals16 & 0xFF) as u8;
        huffman[3] = ((nvals16 >> 8) & 0xFF) as u8;
        huffman[4] = (((nvals16) >> 16) & 0xFF) as u8;
        huffman[5] = (((nvals16) >> 24) & 0xFF) as u8;
        huffman[6] = (nvals & 0xFF) as u8;
        huffman[7] = (((nvals) >> 8) & 0xFF) as u8;
        huffman[8] = (((nvals) >> 16) & 0xFF) as u8;
        huffman[9] = (((nvals) >> 24) & 0xFF) as u8;
        huffman[10] = ((nhuff) & 0xFF) as u8;
        huffman[11] = (((nhuff) >> 8) & 0xFF) as u8;
        huffman[12] = (((nhuff) >> 16) & 0xFF) as u8;
        huffman[13] = (((nhuff) >> 24) & 0xFF) as u8;
        huffman[14 + nhuff] = ((nhuffdict) & 0xFF) as u8;
        huffman[15 + nhuff] = (((nhuffdict) >> 8) & 0xFF) as u8;
        huffman[16 + nhuff] = (((nhuffdict) >> 16) & 0xFF) as u8;
        huffman[17 + nhuff] = ((ndict) & 0xFF) as u8;
        huffman[18 + nhuff] = (((ndict) >> 8) & 0xFF) as u8;
        huffman[19 + nhuff] = (((ndict) >> 16) & 0xFF) as u8;
        for i in 0..nhuffdict {
            huffman[20 + nhuff + i] = huffdict[i];
        }
    } else if (*chosen_algo == 1)
        || ((*chosen_algo == -1) && (huffman_lengths[1] < huffman_lengths[2]))
    {
        *chosen_algo = 1;
        *huffman_len = i32::try_from(huffman_lengths[1]).expect("i32 from usize");
        huffman[0] = isvals16 as u8;
        huffman[1] = 1;
        huffman[2] = ((nvals16) & 0xFF) as u8;
        huffman[3] = (((nvals16) >> 8) & 0xFF) as u8;
        huffman[4] = (((nvals16) >> 16) & 0xFF) as u8;
        huffman[5] = (((nvals16) >> 24) & 0xFF) as u8;
        huffman[6] = ((nvals) & 0xFF) as u8;
        huffman[7] = (((nvals) >> 8) & 0xFF) as u8;
        huffman[8] = (((nvals) >> 16) & 0xFF) as u8;
        huffman[9] = (((nvals) >> 24) & 0xFF) as u8;
        huffman[10] = ((nhuff) & 0xFF) as u8;
        huffman[11] = (((nhuff) >> 8) & 0xFF) as u8;
        huffman[12] = (((nhuff) >> 16) & 0xFF) as u8;
        huffman[13] = (((nhuff) >> 24) & 0xFF) as u8;
        huffman[14 + nhuff] = ((nhuffdictunpack) & 0xFF) as u8;
        huffman[15 + nhuff] = (((nhuffdictunpack) >> 8) & 0xFF) as u8;
        huffman[16 + nhuff] = (((nhuffdictunpack) >> 16) & 0xFF) as u8;
        huffman[17 + nhuff] = ((ndict) & 0xFF) as u8;
        huffman[18 + nhuff] = (((ndict) >> 8) & 0xFF) as u8;
        huffman[19 + nhuff] = (((ndict) >> 16) & 0xFF) as u8;
        huffman[20 + nhuff] = ((nhuff1) & 0xFF) as u8;
        huffman[21 + nhuff] = (((nhuff1) >> 8) & 0xFF) as u8;
        huffman[22 + nhuff] = (((nhuff1) >> 16) & 0xFF) as u8;
        huffman[23 + nhuff] = ((nhuffdict1) & 0xFF) as u8;
        huffman[24 + nhuff] = (((nhuffdict1) >> 8) & 0xFF) as u8;
        huffman[25 + nhuff] = (((nhuffdict1) >> 16) & 0xFF) as u8;
        huffman[26 + nhuff] = ((ndict1) & 0xFF) as u8;
        huffman[27 + nhuff] = (((ndict1) >> 8) & 0xFF) as u8;
        huffman[28 + nhuff] = (((ndict1) >> 16) & 0xFF) as u8;
        for i in 0..nhuff1 {
            huffman[29 + nhuff + i] = huffman1[i];
        }
        for i in 0..nhuffdict1 {
            huffman[29 + nhuff + nhuff1 + i] = huffdict1[i];
        }
    } else {
        *chosen_algo = 2;
        *huffman_len = i32::try_from(huffman_lengths[2]).expect("i32 from usize");
        huffman[0] = isvals16 as u8;
        huffman[1] = 2;
        huffman[2] = ((nvals16) & 0xFF) as u8;
        huffman[3] = (((nvals16) >> 8) & 0xFF) as u8;
        huffman[4] = (((nvals16) >> 16) & 0xFF) as u8;
        huffman[5] = (((nvals16) >> 24) & 0xFF) as u8;
        huffman[6] = ((nvals) & 0xFF) as u8;
        huffman[7] = (((nvals) >> 8) & 0xFF) as u8;
        huffman[8] = (((nvals) >> 16) & 0xFF) as u8;
        huffman[9] = (((nvals) >> 24) & 0xFF) as u8;
        huffman[10] = ((nhuff) & 0xFF) as u8;
        huffman[11] = (((nhuff) >> 8) & 0xFF) as u8;
        huffman[12] = (((nhuff) >> 16) & 0xFF) as u8;
        huffman[13] = (((nhuff) >> 24) & 0xFF) as u8;
        huffman[14 + nhuff] = ((nhuffdictunpack) & 0xFF) as u8;
        huffman[15 + nhuff] = (((nhuffdictunpack) >> 8) & 0xFF) as u8;
        huffman[16 + nhuff] = (((nhuffdictunpack) >> 16) & 0xFF) as u8;
        huffman[17 + nhuff] = ((ndict) & 0xFF) as u8;
        huffman[18 + nhuff] = (((ndict) >> 8) & 0xFF) as u8;
        huffman[19 + nhuff] = (((ndict) >> 16) & 0xFF) as u8;
        huffman[20 + nhuff] = ((nhuffrle) & 0xFF) as u8;
        huffman[21 + nhuff] = (((nhuffrle) >> 8) & 0xFF) as u8;
        huffman[22 + nhuff] = (((nhuffrle) >> 16) & 0xFF) as u8;
        huffman[23 + nhuff] = ((nhuff2) & 0xFF) as u8;
        huffman[24 + nhuff] = (((nhuff2) >> 8) & 0xFF) as u8;
        huffman[25 + nhuff] = (((nhuff2) >> 16) & 0xFF) as u8;
        huffman[26 + nhuff] = ((nhuffdict2) & 0xFF) as u8;
        huffman[27 + nhuff] = (((nhuffdict2) >> 8) & 0xFF) as u8;
        huffman[28 + nhuff] = (((nhuffdict2) >> 16) & 0xFF) as u8;
        huffman[29 + nhuff] = ((ndict2) & 0xFF) as u8;
        huffman[30 + nhuff] = (((ndict2) >> 8) & 0xFF) as u8;
        huffman[31 + nhuff] = (((ndict2) >> 16) & 0xFF) as u8;
        for i in 0..nhuff2 {
            huffman[32 + nhuff + i] = huffman2[i];
        }
        for i in 0..nhuffdict2 {
            huffman[32 + nhuff + nhuff2 + i] = huffdict2[i];
        }
    }
}

fn read3le(data: &[u8], offset: usize) -> i32 {
    (data[offset] as i32) | ((data[offset + 1] as i32) << 8) | ((data[offset + 2] as i32) << 16)
}

pub(crate) fn ptngc_comp_huff_decompress(huffman: &[u8], huffman_len: i32, vals: &mut [u32]) {
    let isvals16 = huffman[0] as i32;
    let algo = huffman[1] as i32;
    let mut nvals16 = i32::from_le_bytes(huffman[2..2 + 4].try_into().expect("error handling"));
    let nvals = i32::from_le_bytes(huffman[6..6 + 4].try_into().expect("error handling"));
    let nhuff = i32::from_le_bytes(huffman[10..10 + 4].try_into().expect("error handling"));
    let ndict = read3le(huffman, 17 + nhuff as usize);

    let mut owner_vals16 = vec![0_u32; nvals16 as usize];
    let mut vals16 = owner_vals16.as_mut_slice();
    if isvals16 != 0 {
        vals16 = vals;
        nvals16 = nvals;
    }

    match algo {
        0 => {
            let nhuffdict = read3le(huffman, 14 + nhuff as usize);
            ptngc_comp_conv_from_huffman(
                &huffman[14..],
                &mut vals16,
                nvals16,
                ndict as usize,
                Some(&huffman[(20 + nhuff) as usize..]),
                None,
            );
        }
        1 => {
            let mut huffdictunpack = vec![0; 0x20005];
            // First the dictionary needs to be uncompressed
            let nhuffdictunpack = read3le(huffman, 14 + nhuff as usize);
            let nhuff1 = read3le(huffman, 20 + nhuff as usize);
            let nhuffdict1 = read3le(huffman, 23 + nhuff as usize);
            let ndict1 = read3le(huffman, 26 + nhuff as usize);
            ptngc_comp_conv_from_huffman(
                &huffman[(29 + nhuff) as usize..],
                &mut huffdictunpack,
                nhuffdictunpack,
                ndict1 as usize,
                Some(&huffman[(29 + nhuff + nhuff1) as usize..]),
                None,
            );
            // Then decompress the "real" data
            ptngc_comp_conv_from_huffman(
                &huffman[14..],
                &mut vals16,
                nvals16,
                ndict as usize,
                None,
                Some(&mut huffdictunpack),
            );
        }
        2 => {
            // let mut huffdictunpack = vec![0; 0x20005];
            let mut huffdictrle = vec![0; 3 * 0x20005 + 3];
            // First the dictionary needs to be uncompressed
            let nhuffdictunpack = read3le(huffman, 14 + nhuff as usize);
            let nhuffrle = read3le(huffman, 20 + nhuff as usize);
            let nhuff2 = read3le(huffman, 23 + nhuff as usize);
            let nhuffdict2 = read3le(huffman, 26 + nhuff as usize);
            let ndict2 = read3le(huffman, 29 + nhuff as usize);
            ptngc_comp_conv_from_huffman(
                &huffman[(32 + nhuff) as usize..],
                &mut huffdictrle,
                nhuffrle,
                ndict2 as usize,
                Some(&huffman[(32 + nhuff + nhuff2) as usize..]),
                None,
            );
            // Then uncompress the rle data
            let mut huffdictunpack =
                ptngc_comp_conv_from_rle(&huffdictrle, nhuffdictunpack as usize);
            // Then decompress the "real" data
            ptngc_comp_conv_from_huffman(
                &huffman[14..],
                &mut vals16,
                nvals16,
                ndict as usize,
                None,
                Some(&mut huffdictunpack),
            );
        }
        _ => unreachable!(),
    }

    // Do I need to convert from vals16?
    if isvals16 == 0 {
        let shadowed_vals16 = vals16.iter().map(|x| *x).collect::<Vec<u32>>();
        let _ = ptngc_comp_conv_from_vals16(&shadowed_vals16, nvals16 as usize, vals);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use super::ptngc_comp_huff_compress_verbose;
    static INIT: Once = Once::new();

    fn init_logger() {
        INIT.call_once(|| {
            env_logger::builder().is_test(true).try_init().ok();
        });
    }

    #[test]
    fn it_works() {
        init_logger();
        let mut vals = vec![1, 2, 3, 4, 5];
        let mut huffman = vec![0; 10000];
        let mut huffman_len = 0;
        let mut huffdatalen = 0;
        let mut huffman_lengths = vec![0; 3];
        let mut chosen_algo = -1;
        let isvals16 = false;

        ptngc_comp_huff_compress_verbose(
            &mut vals,
            &mut huffman,
            &mut huffman_len,
            &mut huffdatalen,
            &mut huffman_lengths,
            &mut chosen_algo,
            isvals16,
        );
        assert_eq!(chosen_algo, 0);
        assert_eq!(huffman_len, 29);
        assert_eq!(huffdatalen, 2);
        assert_eq!(huffman_lengths, vec![29, 41, 45]);
        assert_eq!(
            huffman[..huffman_len as usize],
            vec![
                0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 27, 112, 7, 0, 0, 5, 0, 0, 5, 0, 0, 69,
                20, 81, 198
            ]
        );
    }
    fn test_algorithm_helper(algo: i32, expected_len: i32, expected_output: &[u8]) {
        init_logger();
        let mut vals = vec![10, 20, 30, 40, 50, 60];
        let mut huffman = vec![0; 10000];
        let mut huffman_len = 0;
        let mut huffdatalen = 0;
        let isvals16 = false;
        let mut chosen_algo = algo;
        let mut huffman_lengths = vec![0; 3];

        ptngc_comp_huff_compress_verbose(
            &mut vals,
            &mut huffman,
            &mut huffman_len,
            &mut huffdatalen,
            &mut huffman_lengths,
            &mut chosen_algo,
            isvals16,
        );

        assert_eq!(chosen_algo, algo);
        assert_eq!(huffman_len, expected_len);
        assert_eq!(huffdatalen, 2);
        assert_eq!(huffman_lengths, vec![37, 55, 58]);
        assert_eq!(&huffman[..huffman_len as usize], expected_output);
    }

    #[test]
    fn all_algorithms() {
        let expected_outputs = [
            (
                0,
                37,
                vec![
                    0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 25, 119, 15, 0, 0, 6, 0, 0, 60, 0, 0,
                    0, 34, 0, 68, 0, 140, 1, 24, 2, 48, 4, 96,
                ],
            ),
            (
                1,
                55,
                vec![
                    0, 1, 6, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 25, 119, 64, 0, 0, 6, 0, 0, 10, 0, 0,
                    14, 0, 0, 4, 0, 0, 224, 1, 128, 24, 1, 0, 32, 4, 0, 128, 60, 0, 0, 133, 28, 64,
                    0, 0, 0, 0, 0, 0, 17, 128,
                ],
            ),
            (
                2,
                58,
                vec![
                    0, 2, 6, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 25, 119, 64, 0, 0, 6, 0, 0, 31, 0, 0,
                    9, 0, 0, 15, 0, 0, 6, 0, 0, 240, 110, 66, 228, 44, 133, 144, 178, 22, 62, 0, 0,
                    138, 40, 146, 70, 0, 0, 0, 0, 0, 0, 1, 32,
                ],
            ),
        ];

        for (algo, expected_len, expected_output) in expected_outputs {
            test_algorithm_helper(algo, expected_len, &expected_output);
        }
    }
}
