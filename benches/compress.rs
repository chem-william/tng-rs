use criterion::{Criterion, criterion_group, criterion_main};
use tng_rs::compress::tng_compress_pos;

/// Generate deterministic position data resembling an MD trajectory.
/// Atoms are placed on a grid and given small per-frame displacements
/// via a simple sine pattern (similar to the C testsuite generators).
fn gen_positions(natoms: usize, nframes: usize, scale: f64) -> Vec<f64> {
    let nitems = natoms * nframes * 3;
    let mut pos = vec![0.0f64; nitems];
    let side = (natoms as f64).cbrt().ceil() as usize;
    for frame in 0..nframes {
        for i in 0..natoms {
            let base = frame * natoms * 3 + i * 3;
            let gx = (i % side) as f64;
            let gy = ((i / side) % side) as f64;
            let gz = (i / (side * side)) as f64;
            let wobble = ((frame * 7 + i) as f64 * 0.1).sin() * 0.02;
            pos[base] = (gx + wobble) * scale;
            pos[base + 1] = (gy + wobble) * scale;
            pos[base + 2] = (gz + wobble) * scale;
        }
    }
    pos
}

fn bench_compress(c: &mut Criterion) {
    // Small system: 1000 atoms, 10 frames (mirrors default test params)
    // algo: TRIPLET_INTRA(3) initial, TRIPLET_INTER(1) coding
    let small_pos = gen_positions(1000, 10, 0.1);
    c.bench_function("compress_small_1k_10f", |b| {
        b.iter(|| {
            let mut algo = [3, -1, 1, 0];
            tng_compress_pos::<f64>(&small_pos, 1000, 10, 0.01, 5, &mut algo)
        });
    });

    // Large system: 100_000 atoms, 1 frame (mirrors test41 params)
    // algo: TRIPLET_INTRA(3) initial only (single frame)
    let large_pos = gen_positions(100_000, 1, 0.5);
    c.bench_function("compress_large_100k_1f", |b| {
        b.iter(|| {
            let mut algo = [3, -1, 1, -1];
            tng_compress_pos::<f64>(&large_pos, 100_000, 1, 1e-8, 5, &mut algo)
        });
    });
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);
