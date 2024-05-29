#![feature(portable_simd)]

use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::mem::MaybeUninit;
use std::simd::num::SimdFloat;
use std::simd::{f64x64, StdFloat};

#[allow(non_upper_case_globals)]
const bench_size: usize = 1000000;

const SIMD_COUNT: usize = bench_size / 64;

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

fn softmax_simd(x: &[f64x64; SIMD_COUNT]) -> [f64x64; SIMD_COUNT] {
    let max: f64 = *x
        .map(|v| v.reduce_max())
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = f64x64::splat(max);

    let x_exp = x.map(|v| (v - max).exp());
    let x_exp_sum = x_exp.iter().map(|v| v.reduce_sum()).sum();
    let divide_by = f64x64::splat(x_exp_sum);
    let probs = x_exp.map(|v| v / divide_by);

    probs
}

fn save_results(times: &[f64], fname: &str) {
    let mut times_diffed = vec![0f64; times.len() - 1];
    for i in 1..times.len() {
        times_diffed[i - 1] = times[i] - times[i - 1];
    }
    let times = times_diffed;

    let filename = format!("{}_{}_rs.json", fname, bench_size);
    let bench_id = std::env::args().nth(2).unwrap();
    let path = std::path::Path::new(".")
        .join("bench_times")
        .join(bench_id)
        .join(filename);

    let mean = times.iter().sum::<f64>() / bench_size as f64;
    let file_str = fname.to_string();
    let times = times
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<String>>()
        .join(",");
    let json = format!(
        r#"{{
            "mean": {},
            "warmup_time": {},
            "bench_time": {},
            "file": "{}",
            "bench_size": {},
            "times": [
                {}
            ]
        }}"#,
        mean, 0, 0, file_str, bench_size, times
    );

    std::fs::write(path, json).unwrap();
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
    #[allow(non_upper_case_globals)]
    const count: usize = 1000;
    let mut times = [0f64; count];
    for i in 0..count {
        black_box(softmax(&arr));
        times[i] = start.elapsed().as_nanos() as f64;
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time (native): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );

    // prepare array of simd vectors

    let mut simd_arr: [f64x64; SIMD_COUNT] = unsafe {
        #[allow(invalid_value)]
        MaybeUninit::uninit().assume_init()
    };

    for i in 0..SIMD_COUNT {
        simd_arr[i] = f64x64::from_slice(&arr[i * 64..]);
    }

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    let mut times_simd = [0f64; count];
    for i in 0..count {
        black_box(softmax_simd(&simd_arr));
        times_simd[i] = start.elapsed().as_nanos() as f64;
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time   (SIMD): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );

    save_results(&times, "softmax_native");
    save_results(&times_simd, "softmax_simd");
}

fn test() {
    // TODO: implement, noop for now
}
