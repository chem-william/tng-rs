// Multiply two 32 bit unsigned integers returning a 64 bit unsigned value (in two integers)
#[inline]
fn ptngc_widemul(i1: u32, i2: u32) -> (u32, u32) {
    let result = (i1 as u64) * (i2 as u64);
    let olo = result as u32;
    let ohi = (result >> 32) as u32;
    (ohi, olo)
}

#[cfg(test)]
mod tests {
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
    fn edge1() {
        let (upper, lower) = ptngc_widemul(1 << 31, 1 << 31);
        assert_eq!(upper, 1073741824);
        assert_eq!(lower, 0);
    }

    #[test]
    fn edge2() {
        let (upper, lower) = ptngc_widemul(1 << 16, (1 << 16) - 1);
        assert_eq!(upper, 0);
        assert_eq!(lower, 4294901760);
    }

    #[test]
    fn edge3() {
        let (upper, lower) = ptngc_widemul(4294967295, 4294967295);
        assert_eq!(upper, 4294967294);
        assert_eq!(lower, 1);
    }
}
