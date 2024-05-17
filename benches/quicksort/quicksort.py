import sys
import time

import numpy as np

bench_size = 10000


def quicksort(data, left, right):
    if left >= right:
        return

    pivot = data[right]
    i = left - 1

    for j in range(left, right):
        if data[j] <= pivot:
            i = i + 1
            (data[i], data[j]) = (data[j], data[i])

    (data[i + 1], data[right]) = (data[right], data[i + 1])

    i += 1

    quicksort(data, left, i - 1)
    quicksort(data, i + 1, right)


def bench_quicksort(data):
    # copy
    test_data = data.copy()
    quicksort(test_data, 0, len(test_data) - 1)


def test():
    data = [3, 6, 8, 10, 1, 2, 1]
    quicksort(data, 0, len(data) - 1)
    assert data == [1, 1, 2, 3, 6, 8, 10]

    # also test 4 3 2 1
    data = [4, 3, 2, 1]
    quicksort(data, 0, len(data) - 1)
    assert data == [1, 2, 3, 4]


def initialize():
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(bench_size,), dtype=np.uint8)
    return arr


def main():
    test()

    arr = initialize()

    # warm up
    test_arr = arr.copy()
    quicksort(test_arr, 0, len(arr) - 1)

    times = 10
    # bench
    start = time.time()
    for i in range(times):
        test_arr = arr.copy()
        quicksort(test_arr, 0, len(test_arr) - 1)

    end = time.time()
    print(f"Mean time: {((end - start) / times)*1000}ms")


if __name__ == "__main__":
    main()
