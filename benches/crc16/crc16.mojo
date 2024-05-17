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


fn main() raises:
    test()
    # size is 1 arg from sys
    var arr = stack_allocation[bench_size, type]()
    # seed(1)
    rand(arr, bench_size)

    var py = Python.import_module("builtins")
    # _ = py.print(py.str("Starting benchmark, size {}...").format(size))

    var crc = crc16_naive[0x8408, bench_size](arr)

    @always_inline
    @parameter
    fn worker():
        var bres = crc16_naive[0x8408, bench_size](arr)
        benchmark.keep(bres)  # do not optimize out

    var r = benchmark.run[worker](max_runtime_secs=5)

    # _ = py.print(py.str("Result: {}, iters: {}").format(res, iters))

    py.print(py.str("Mean time: {}ms").format(r.mean("ms")))
