// Multiply two 32 bit unsigned integers returning a 64 bit unsigned value (in two integers)
fn ptngc_widemul(i1: u32, i2: u32) -> (u32, u32) {
    let result = (i1 as u64) * (i2 as u64);
    let olo = result as u32;
    let ohi = (result >> 32) as u32;
    (ohi, olo)
}

// Divide a 64 bit unsigned value in hi:lo with the 32 bit value i and return (result, remainder)
fn ptngc_widediv(hi: u32, lo: u32, i: u32) -> (u32, u32) {
    let v = (hi as u64) << 32 | lo as u64;
    let res = v / (i as u64);
    let rem = v - res * (i as u64);

    let result = res as u32;
    let remainder = rem as u32;

    (result, remainder)
}

// Add a u32 to a `largeint`. `j` determines which value in the `largeint` to add `v1` to.
#[inline]
fn largeint_add_gen(v1: u32, largeint: &mut [u32], n: usize, j: usize) {
    let mut tmp_j = j;
    if tmp_j >= n || tmp_j >= largeint.len() {
        return;
    }

    let (sum, mut carry) = largeint[tmp_j].overflowing_add(v1);
    largeint[tmp_j] = sum;

    tmp_j += 1;
    while tmp_j < n && carry {
        let (new_sum, new_carry) = largeint[tmp_j].overflowing_add(1);
        largeint[tmp_j] = new_sum;
        carry = new_carry;
        tmp_j += 1;
    }
}

// Add a `u32` to a `largeint`
pub(crate) fn ptngc_largeint_add(v1: u32, largeint: &mut [u32], n: usize) {
    largeint_add_gen(v1, largeint, n, 0);
}

pub(crate) fn ptngc_largeint_mul(v1: u32, largeint_in: &[u32], largeint_out: &mut [u32], n: usize) {
    largeint_out.fill(0);

    let mut i = 0;
    while i < n - 1 {
        if largeint_in[i] != 0 {
            let (hi, lo) = ptngc_widemul(v1, largeint_in[i]);
            largeint_add_gen(lo, largeint_out, n, i);
            largeint_add_gen(hi, largeint_out, n, i + 1);
        }
        i += 1;
    }
    if largeint_in[i] != 0 {
        let (_, lo) = ptngc_widemul(v1, largeint_in[i]); // 32x32->64 mul
        largeint_add_gen(lo, largeint_out, n, i);
    }
}

// Return the remainder from dividing largeint_in with v1. Result of the division is returned in largeint_out
pub(crate) fn ptngc_largeint_div(
    v1: u32,
    largeint_in: &[u32],
    largeint_out: &mut [u32],
    n: usize,
) -> u32 {
    let mut remainder = 0;

    // Boot
    let mut hi = 0;
    let mut i = n;
    while i != 0 {
        i -= 1;
        let (result, remainder_tmp) = ptngc_widediv(hi, largeint_in[i], v1);
        remainder = remainder_tmp;
        largeint_out[i] = result;
        hi = remainder;
    }
    remainder
}
