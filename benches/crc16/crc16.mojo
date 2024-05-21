import benchmark
import sys
from random import rand, seed
from python import Python

alias type = DType.int8
alias bench_size = 1000000


@always_inline
fn crc16_naive[poly: Int, len: Int](data: DTypePointer[type]) -> Int:
    # CRC-16-CCITT Algorithm
    # naively ported from python version

    var crc = 0xFFFF

    for b in range(len):
        var cur_byte = 0xFF & data[b]

        @unroll
        for _ in range(8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1

    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


fn test() raises:
    # s = b"\x31\x32\x33\x34\x35\x36\x37\x38\x39"
    var s: StringLiteral = "123456789"
    var data: DTypePointer[type] = s.data()

    # print as asci
    var crc = crc16_naive[0x8408, 9](data)

    if not crc == 0x6E90:
        raise "Test failed"


fn initialize() -> DTypePointer[type]:
    var arr = DTypePointer[type].alloc(bench_size)
    # seed(1)
    rand(arr, bench_size)
    return arr


fn bench(
    bench_id: StringRef, bench_time: Int = 5
) raises -> Dict[StringRef, benchmark.Report]:
    test()
    var arr = initialize()

    var crc = crc16_naive[0x8408, bench_size](arr)

    @always_inline
    @parameter
    fn worker():
        var bres = crc16_naive[0x8408, bench_size](arr)
        benchmark.keep(bres)  # do not optimize out

    var report = benchmark.run[worker](
        min_runtime_secs=bench_time, max_runtime_secs=bench_time
    )
    var res = Dict[StringRef, benchmark.Report]()
    res["crc16_naive"] = report
    return res


fn main() raises:
    var py = Python.import_module("builtins")
    var reports = bench("crc16_naive", 5)
    var report = reports["crc16_naive"]
    py.print(py.str("Mean time: {}ms").format(report.mean("ms")))
