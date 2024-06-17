import math
import sys
import time
import numpy as np

bench_size = 128


def bench_matmul_np(pair):
    a, b = pair
    return np.matmul(a, b)


native_pair = None


def bench_matmul_native(pair):

    a, b = pair
    m, n = len(a), len(a[0])
    p = len(b[0])

    # init c
    c = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]

    return c


def test():
    x = [[1, 2], [3, 4]]
    y = [[5, 6], [7, 8]]

    x_np = np.array(x)
    y_np = np.array(y)

    res = bench_matmul_native((x, y))
    res_np = bench_matmul_np((x_np, y_np))

    assert np.allclose(np.array(res), res_np)
    assert np.allclose(res, [[19, 22], [43, 50]])


def initialize():
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 1.0, (bench_size, bench_size))
    b = rng.uniform(0.0, 1.0, (bench_size, bench_size))
    return (a, b)


def main():
    test()

    rng = np.random.default_rng(42)
    numpy_in_a, numpy_in_b = initialize()
    native_in_a = numpy_in_a.tolist()
    native_in_b = numpy_in_b.tolist()

    # warm up
    bench_matmul_native((native_in_a, native_in_b))
    bench_matmul_np((numpy_in_a, numpy_in_b))

    times = 20
    # bench
    start = time.time()
    for i in range(times):
        bench_matmul_native((native_in_a, native_in_b))
    end = time.time()

    print(f"Mean time (native): {((end - start) / times)*1000}ms")

    start = time.time()
    for i in range(times):
        bench_matmul_np((numpy_in_a, numpy_in_b))
    end = time.time()

    print(f"Mean time  (numpy): {((end - start) / times)*1000}ms")


if __name__ == "__main__":
    main()
