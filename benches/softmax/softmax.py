import math
import sys
import time
import numpy as np

bench_size = 10000


def softmax_np(x):
    maxes = np.max(x, axis=1, keepdims=True)[0]
    x_exp = np.exp(x - maxes)
    x_exp_sum = np.sum(x_exp, 1, keepdims=True)
    probs = x_exp / x_exp_sum
    return probs


def softmax_native(x: list[float]) -> list[float]:
    maxes = max(x)
    x_exp = [math.exp(xi - maxes) for xi in x]
    x_exp_sum = sum(x_exp)
    probs = [xi / x_exp_sum for xi in x_exp]
    return probs


def test():
    x = [1, 2, 3]
    assert softmax_native(x) == list(softmax_np(np.array([x]))[0])

    x = [1, 2, 3, 4]
    assert softmax_native(x) == list(softmax_np(np.array([x]))[0])


def main():
    test()

    rng = np.random.default_rng(42)
    numpy_in = rng.random((bench_size, 1))
    native_in = numpy_in.flatten().tolist()

    # warm up
    softmax_native(native_in)
    softmax_np(numpy_in)

    times = 1000
    # bench
    start = time.time()
    for i in range(times):
        softmax_native(native_in)
    end = time.time()

    print(f"Mean time (native): {((end - start) / times)*1000}ms")

    start = time.time()
    for i in range(times):
        softmax_np(numpy_in)
    end = time.time()

    print(f"Mean time (numpy): {((end - start) / times)*1000}ms")


if __name__ == "__main__":
    main()
