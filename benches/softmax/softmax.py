import math
import sys
import time
import numpy as np

bench_size = 10000


def bench_softmax_np(x):
    maxes = np.max(x, axis=1, keepdims=True)[0]
    x_exp = np.exp(x - maxes)
    x_exp_sum = np.sum(x_exp, 1, keepdims=True)
    probs = x_exp / x_exp_sum
    return probs


def bench_softmax_native(x: list[float]) -> list[float]:
    maxes = max(x)
    x_exp = [math.exp(xi - maxes) for xi in x]
    x_exp_sum = sum(x_exp)
    probs = [xi / x_exp_sum for xi in x_exp]
    return probs


def test():
    x = [1.0, 2.0, 3.0]
    assert np.allclose(
        np.array(bench_softmax_native(x)), bench_softmax_np(np.array([x]))
    )

    x = [1.0, 2.0, 3.0, 4.0]
    assert np.allclose(
        np.array(bench_softmax_native(x)), bench_softmax_np(np.array([x]))
    )


def initialize():
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, (bench_size, 1))


def main():
    test()

    numpy_in = initialize()
    native_in = numpy_in.flatten().tolist()

    # warm up
    bench_softmax_native(native_in)
    bench_softmax_np(numpy_in)

    times = 1000
    # bench
    start = time.time()
    for i in range(times):
        bench_softmax_native(native_in)
    end = time.time()

    print(f"Mean time (native): {((end - start) / times)*1000}ms")

    start = time.time()
    for i in range(times):
        bench_softmax_np(numpy_in)
    end = time.time()

    print(f"Mean time  (numpy): {((end - start) / times)*1000}ms")


if __name__ == "__main__":
    main()
