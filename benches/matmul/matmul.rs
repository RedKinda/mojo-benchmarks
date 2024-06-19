use std::fs::File;
use std::hint::black_box;
use std::io::Read;

#[allow(non_upper_case_globals)]
const bench_size: usize = 128;

fn matmul<const M: usize, const N: usize, const P: usize>(
    pair: (&[f64], &[f64]),
    result: &mut [f64],
) {
    let (a, b) = pair;

    for i in 0..M {
        for j in 0..P {
            for k in 0..N {
                result[i * P + j] += a[i * P + k] * b[k * P + j];
            }
        }
    }
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
    let bench_time = std::env::args().nth(3).unwrap().parse::<usize>().unwrap();
    let bench_time_ns: f64 = bench_time as f64 * 1_000_000_000f64;
    test();
    // arr is u8 1000000 of random elements allocated on heap
    let mut arr = Box::new([0f64; bench_size * bench_size]);
    let mut arr2 = Box::new([0f64; bench_size * bench_size]);
    let mut f = File::open("/dev/urandom").unwrap();
    let mut buf = Box::new([0u8; bench_size * bench_size * 8]);
    let mut buf2 = Box::new([0u8; bench_size * bench_size * 8]);
    f.read_exact(&mut *buf).unwrap();
    f.read_exact(&mut *buf2).unwrap();

    // uniform random u8 -> f64
    for i in buf.chunks_exact(8) {
        let mut num = 0u64;
        for j in 0..8 {
            num |= (i[j] as u64) << (j * 8);
        }
        let num = num as f64 / f64::MAX;
        arr[i.len() / 8] = num;
    }

    for i in buf2.chunks_exact(8) {
        let mut num = 0u64;
        for j in 0..8 {
            num |= (i[j] as u64) << (j * 8);
        }
        let num = num as f64 / f64::MAX;
        arr2[i.len() / 8] = num;
    }

    // println!("starting..");

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    let mut times = vec![];
    let mut res = Box::new([0f64; bench_size * bench_size]);

    loop {
        // zero out res
        res.fill(0.0);
        black_box(matmul::<bench_size, bench_size, bench_size>(
            (&*arr, &*arr2),
            &mut *res,
        ));
        let time = start.elapsed().as_nanos() as f64;
        if time > bench_time_ns {
            times.push(time);
            break;
        }
        times.push(time);
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time: {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / times.len() as f64
    );

    save_results(&times, "matmul");
}

fn test() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [5.0, 6.0, 7.0, 8.0];

    let mut res = [0.0; 4];

    matmul::<2, 2, 2>((&x, &y), &mut res);

    assert_eq!(res, [19.0, 22.0, 43.0, 50.0]);
}
