use std::fs::File;
use std::hint::black_box;
use std::io::Read;

#[allow(non_upper_case_globals)]
const bench_size: usize = 1000000;

fn crc16(data: &[u8]) -> u16 {
    let poly = 0x8408;
    let mut crc: u16 = 0xffff;

    for byte in data.iter() {
        let mut cur_byte = *byte as u16 & 0xff;
        for _ in 0..8 {
            if (crc & 0x0001) ^ (cur_byte & 0x0001) != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
            cur_byte >>= 1;
        }
    }

    crc = !crc & 0xffff;
    crc = (crc << 8) | ((crc >> 8) & 0xff);
    crc
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
    let bench_time = std::env::args().nth(3).unwrap().parse::<usize>().unwrap();
    let bench_time_ns: f64 = bench_time as f64 * 1_000_000_000f64;
    test();
    // arr is u8 1000000 of random elements allocated on heap
    let mut arr = vec![0u8; bench_size];
    let mut f = File::open("/dev/urandom").unwrap();
    f.read_exact(&mut arr).unwrap();

    crc16(black_box(&arr));

    // println!("starting..");

    let start = std::time::Instant::now();
    let mut times = vec![];

    loop {
        black_box(crc16(&arr));
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
        elapsed as f64 / times.len() as f64 * 1_000_000f64
    );

    save_results(&times, "crc16");
}

fn test() {
    // b"\x31\x32\x33\x34\x35\x36\x37\x38\x39"
    let s = &[0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39];
    assert_eq!(crc16(s), 0x6e90);
}
