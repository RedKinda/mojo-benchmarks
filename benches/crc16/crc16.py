import sys
import numpy as np

import time

bench_size = 100000

# Adapted from https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6
def bench_crc16(data, poly=0x8408):
    """
    CRC-16-CCITT Algorithm
    """

    crc = 0xFFFF
    for b in data:
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


def initialize():
    from numpy.random import default_rng

    rng = default_rng(42)
    data = rng.integers(0, 256, size=(bench_size,), dtype=np.uint8)
    return data


def test():

    # fn test() {
    #     let s = "12345678".as_bytes();
    #     assert_eq!(crc16(&s), 50389);
    # }
    s = b"\x31\x32\x33\x34\x35\x36\x37\x38\x39"
    # print as ascii
    crc = bench_crc16(s)
    assert crc == 0x6E90


def main():
    test()

    arr = initialize()

    # warm up
    bench_crc16(arr)

    # print("CRC16: ", crc16(arr))
    times = 10
    # bench
    start = time.time()
    for i in range(times):
        bench_crc16(arr)

    end = time.time()
    print(f"Mean time: { ((end - start) / times)*1000}ms")


if __name__ == "__main__":
    main()
