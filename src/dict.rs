// In your lib.rs or a module file:

/// Size of the full canonical dictionary / histogram table.
pub(crate) const DICT_SIZE: usize = 0x20004;

/// Fill `dict` with the canonical sequence `[0, 1, 2, …, DICT_SIZE−1]`.
///
/// # Panic
///
/// Panics if `dict.len() < DICT_SIZE`.
pub(crate) fn ptngc_comp_canonical_dict(dict: &mut [u32]) {
    assert!(dict.len() >= DICT_SIZE, "dict buffer too small");
    for (i, slot) in dict.iter_mut().enumerate().take(DICT_SIZE) {
        *slot = i as u32;
    }
}

/// Build a “dictionary” and histogram from the input values:
/// - `vals`: input symbols in `[0..DICT_SIZE)`.
/// - Returns `(dict, hist)`, where:
///     * `hist[i]` = count of symbol `i` in `vals`.
///     * `dict[..ndict]` = the list of symbols that occurred (in ascending order).
pub fn ptngc_comp_make_dict_hist(vals: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut j = 0;
    let nvals = vals.len();
    // hist[i] counts occurrences of symbol i
    let mut hist = vec![0u32; DICT_SIZE];
    for &v in vals {
        let idx = v as usize;
        hist[idx] = hist[idx].saturating_add(1);
    }

    // dict: symbols with nonzero count
    let mut dict = Vec::with_capacity(nvals);
    for i in 0..DICT_SIZE {
        if hist[i] != 0 {
            hist[j] = hist[i];
            dict.push(u32::try_from(i).expect("u32 from usize"));
            j += 1;
            if j == nvals {
                break;
            }
        }
    }

    (dict[..j].into(), hist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_dict() {
        let mut dict = vec![0; DICT_SIZE];
        ptngc_comp_canonical_dict(&mut dict);

        // spot-check start, middle, end
        assert_eq!(dict[0], 0);
        assert_eq!(dict[1], 1);
        assert_eq!(dict[256], 256);
        assert_eq!(dict[DICT_SIZE - 1], (DICT_SIZE - 1) as u32);
    }

    #[test]
    fn make_dict_hist_empty() {
        let vals: Vec<u32> = vec![];
        let (dict, hist) = ptngc_comp_make_dict_hist(&vals);
        assert_eq!(dict.len(), 0);
        assert!(hist.iter().all(|&c| c == 0));
    }

    #[test]
    fn make_dict_hist_single() {
        let vals = vec![123];
        let (dict, hist) = ptngc_comp_make_dict_hist(&vals);
        assert_eq!(dict, vec![123]);
        assert_eq!(hist[123], 1);
    }

    #[test]
    fn make_dict_hist_multiple() {
        let vals = vec![3, 1, 3, 7, 1];
        let (dict, hist) = ptngc_comp_make_dict_hist(&vals);
        // symbols seen in ascending order: 1, 3, 7
        assert_eq!(dict, vec![1, 3, 7]);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[3], 2);
        assert_eq!(hist[7], 1);
        // the rest remain zero
        for (i, &c) in hist.iter().enumerate() {
            if i == 1 || i == 3 || i == 7 {
                continue;
            }
        }
    }

    #[test]
    fn make_dict_hist_limit_nvals() {
        // if more distinct symbols than nvals, we stop at nvals entries
        let vals: Vec<u32> = (0..10).collect();
        let (dict, _) = ptngc_comp_make_dict_hist(&vals);
        assert_eq!(dict.len(), 10);
        assert_eq!(dict, (0..10).collect::<Vec<_>>());
    }
}
