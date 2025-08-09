// Multiply two 32 bit unsigned integers returning a 64 bit unsigned value (in two integers)
fn ptngc_widemul(i1: u32, i2: u32) -> (u32, u32) {
    let result = (i1 as u64) * (i2 as u64);
    let olo = result as u32;
    let ohi = (result >> 32) as u32;
    (ohi, olo)
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

// Add a `u32` to a `largeint`
pub(crate) fn ptngc_largeint_add(v1: u32, largeint: &mut [u32], n: usize) {
    largeint_add_gen(v1, largeint, n, 0);
}

#[cfg(test)]
mod widemul_tests {
    use super::*;

    #[test]
    fn it_works() {
        let (upper, lower) = ptngc_widemul(32, 32);
        assert_eq!(upper, 0);
        assert_eq!(lower, 1024);
    }

    #[test]
    fn one_upper() {
        let (upper, lower) = ptngc_widemul(0x10000, 0x10000);
        assert_eq!(upper, 1);
        assert_eq!(lower, 0);
    }

    #[test]
    fn u16_mul_u16_minus_one() {
        let (upper, lower) = ptngc_widemul(1 << 16, (1 << 16) - 1);
        assert_eq!(upper, 0);
        assert_eq!(lower, 4294901760);
    }

    #[test]
    fn large_numbers() {
        // 4294967295 = 1<<32 - 1
        let (upper, lower) = ptngc_widemul(4294967295, 4294967295);
        assert_eq!(upper, 4294967294);
        assert_eq!(lower, 1);
    }

    #[test]
    fn gives_zeros() {
        let v1 = 0;
        let largeint_in = vec![1, 2, 3, 4];
        let mut largeint_out = vec![0; 4];

        ptngc_largeint_mul(v1, &largeint_in, &mut largeint_out, 4);

        assert_eq!(largeint_out, [0; 4]);
    }

    #[test]
    fn multiply_by_one() {
        let v1 = 1;
        let largeint_in = vec![1, 2, 3, 4];
        let mut largeint_out = vec![0; 4];

        ptngc_largeint_mul(v1, &largeint_in, &mut largeint_out, 4);

        assert_eq!(largeint_out, [1, 2, 3, 4]);
    }

    #[test]
    fn simple_multiplication() {
        let v1 = 3;
        let largeint_in = vec![2, 0, 0, 0];
        let mut largeint_out = vec![0; 4];

        ptngc_largeint_mul(v1, &largeint_in, &mut largeint_out, 4);

        assert_eq!(largeint_out, [6, 0, 0, 0]);
    }

    #[test]
    // Multiplication causes overflow into next position
    fn multiplication_with_carry() {
        let largeint_in = [0x80000000, 0, 0, 0]; // 2^31
        let mut largeint_out = [0; 4];
        ptngc_largeint_mul(2, &largeint_in, &mut largeint_out, 4);
        // 2 * 2^31 = 2^32 = 0x100000000
        assert_eq!(largeint_out, [0, 1, 0, 0]); // Overflow to next position
    }

    #[test]
    fn max_value_multiplication() {
        let largeint_in = [0xFFFFFFFF, 0, 0, 0];
        let mut largeint_out = [0; 4];
        ptngc_largeint_mul(2, &largeint_in, &mut largeint_out, 4);
        // 2 * 0xFFFFFFFF = 0x1FFFFFFFE
        assert_eq!(largeint_out, [0xFFFFFFFE, 1, 0, 0]);
    }

    #[test]
    // Multiple carries propagate through the array
    fn carry_propagation() {
        let largeint_in = [0xFFFFFFFF, 0xFFFFFFFF, 0, 0];
        let mut largeint_out = [0; 4];
        ptngc_largeint_mul(2, &largeint_in, &mut largeint_out, 4);
        // This should cause carries to propagate
        assert_eq!(largeint_out, [0xFFFFFFFE, 0xFFFFFFFF, 1, 0]);
    }

    #[test]
    // Overflow from last element is handled correctly (discarded)
    fn last_element_overflow() {
        let largeint_in = [0, 0, 0, 0xFFFFFFFF];
        let mut largeint_out = [0; 4];
        ptngc_largeint_mul(2, &largeint_in, &mut largeint_out, 4);
        // The high part of the multiplication should be discarded
        assert_eq!(largeint_out, [0, 0, 0, 0xFFFFFFFE]);
    }

    #[test]
    fn complex_multiplication() {
        let largeint_in = [0x12345678, 0x9ABCDEF0, 0, 0];
        let mut largeint_out = [0; 4];
        ptngc_largeint_mul(0x1000, &largeint_in, &mut largeint_out, 4);

        assert_eq!(largeint_out, [1164410880, 3454992675, 2475, 0,]);
    }

    #[test]
    fn large_multiplier() {
        let largeint_in = [1, 0, 0, 0];
        let mut largeint_out = [0; 4];
        ptngc_largeint_mul(0xFFFFFFFF, &largeint_in, &mut largeint_out, 4);
        assert_eq!(largeint_out, [0xFFFFFFFF, 0, 0, 0]);
    }
}
