#![feature(portable_simd)]

use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::mem::MaybeUninit;
use std::simd::num::SimdFloat;
use std::simd::{f64x2, StdFloat};

#[allow(non_upper_case_globals)]
const bench_size: usize = 1000000;

const SIMD_COUNT: usize = bench_size / 64;

// Set the SIMD type
type SimdType = f64x2;

fn softmax(x: &[f64; bench_size]) -> [f64; bench_size] {
    let mut max = x[0];
    for i in 1..x.len() {
        if x[i] > max {
            max = x[i];
        }
    }

    let mut x_exp = [0.0; bench_size];
    let mut x_exp_sum = 0.0;
    for i in 0..x.len() {
        x_exp[i] = (x[i] - max).exp();
        x_exp_sum += x_exp[i];
    }

    let mut probs = [0.0; bench_size];
    for i in 0..x.len() {
        probs[i] = x_exp[i] / x_exp_sum;
    }

    probs
}

fn softmax_simd(x: &[SimdType; SIMD_COUNT]) -> [SimdType; SIMD_COUNT] {
    let max = x
        .iter()
        .fold(SimdType::splat(f64::NEG_INFINITY), |acc, v| {
            acc.simd_max(*v)
        })
        .reduce_max();
    let max = SimdType::splat(max);

    let x_exp = x.map(|v| (v - max).exp());
    let x_exp_sum = x_exp
        .iter()
        .fold(SimdType::splat(0.0), |acc, v| acc + v)
        .reduce_sum();
    let divide_by = SimdType::splat(x_exp_sum);

    x_exp.map(|v| v / divide_by)
}

fn main() {
    test();
    let mut random_arr = [0u8; bench_size * 8];
    let mut f = File::open("/dev/urandom").unwrap();
    f.read_exact(&mut random_arr).unwrap();

    let mut arr = [0f64; bench_size];
    // convert random bytes to f64
    for i in 0..bench_size {
        arr[i] = f64::from_le_bytes([
            random_arr[i * 8],
            random_arr[i * 8 + 1],
            random_arr[i * 8 + 2],
            random_arr[i * 8 + 3],
            random_arr[i * 8 + 4],
            random_arr[i * 8 + 5],
            random_arr[i * 8 + 6],
            random_arr[i * 8 + 7],
        ]);
    }

    softmax(black_box(&arr));

    // println!("starting..");

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    let count = 1000;
    for _ in 0..count {
        black_box(softmax(&arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time (native): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );

    // prepare array of simd vectors

    let mut simd_arr: [SimdType; SIMD_COUNT] = unsafe {
        #[allow(invalid_value)]
        MaybeUninit::uninit().assume_init()
    };

    for i in 0..SIMD_COUNT {
        simd_arr[i] = SimdType::from_slice(&arr[i * 64..]);
    }

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    for _ in 0..count {
        black_box(softmax_simd(&simd_arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time   (SIMD): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );
}

fn test() {
    // TODO: implement, noop for now
}
