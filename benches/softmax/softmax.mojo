import benchmark
import sys
from random import rand, seed
from python import Python
import math
from algorithm import vectorize

alias type = DType.float64
alias bench_size = 2048
alias _simd_width = 16

"""

def softmax_native(x: list[float]) -> list[float]:
    maxes = max(x)
    x_exp = [math.exp(xi - maxes) for xi in x]
    x_exp_sum = sum(x_exp)
    probs = [xi / x_exp_sum for xi in x_exp]
    return probs
"""


fn softmax[
    size: Int
](x: DTypePointer[type], res_ptr: DTypePointer[type]) -> DTypePointer[type]:
    var max = x[0]
    for i in range(1, size):
        if x[i] > max:
            max = x[i]

    var x_exp = stack_allocation[size, type]()
    var x_exp_sum = 0.0
    for i in range(size):
        x_exp[i] = math.exp(x[i] - max)
        x_exp_sum += x_exp[i]

    for i in range(size):
        res_ptr[i] = x_exp[i] / x_exp_sum

    return res_ptr


fn softmax_simd[size: Int](x: SIMD[type, size]) -> SIMD[type, size]:
    var max = x.reduce_max()

    var x_exp = math.exp(x - max)
    var x_sum = x_exp.reduce_add()

    var probs = x_exp / x_sum

    return probs


fn softmax_simd_proper[
    size: Int, simd_width: Int
](inp: DTypePointer[type], res: DTypePointer[type]) -> DTypePointer[type]:
    var max = inp[0]

    @parameter
    fn closure_max[simd_width: Int](i: Int):
        max = math.max(max, inp.load[width=simd_width](i).reduce_max())

    vectorize[closure_max, simd_width, size=size]()

    var sum = 0.0

    var x_exp = stack_allocation[size, type]()

    @parameter
    fn closure_exp[simd_width: Int](i: Int):
        var x = inp.load[width=simd_width](i)
        var exp = math.exp(x - max)
        sum += exp.reduce_add()
        x_exp.store[width=simd_width](i, exp)

    vectorize[closure_exp, simd_width, size=size]()

    @parameter
    fn closure_div[simd_width: Int](i: Int):
        var x = x_exp.load[width=simd_width](i)
        var prob = x / sum
        res.store[width=simd_width](i, prob)

    vectorize[closure_div, simd_width, size=size]()

    return res


def test():
    # import python softmax.py and compare with result of softmax_native(x)
    Python.add_to_path(".")
    Python.add_to_path("benches/softmax")
    var pysoftmax = Python.import_module("softmax")
    var py = Python.import_module("builtins")
    var pyrandom = Python.import_module("random")

    alias testsize = 16

    var mojoin = stack_allocation[testsize, type]()
    var res_mojo = stack_allocation[testsize, type]()

    pyin = py.list()
    for i in range(testsize):
        var val = pyrandom.random()
        pyin.append(val)
        mojoin[i] = val.to_float64()

    var res_py = pysoftmax.bench_softmax_native(pyin)
    var _a = softmax[testsize](mojoin, res_mojo)
    var res_simd_mojo = softmax_simd[testsize](mojoin.load[width=testsize](0))
    var res_simd_proper_mojo = softmax_simd_proper[testsize, 2](
        mojoin, stack_allocation[testsize, type]()
    )

    for i in range(testsize):
        # acceptable error margin due to float precision
        if math.abs(res_py[i].to_float64() - res_mojo[i]) > 1e-6:
            py.print(py.str("Mismatch at index {}").format(i))
            py.print(
                py.str("Python: {}, Mojo: {}").format(
                    res_py[i].to_float64(), res_mojo[i]
                )
            )
            raise "Test fail"

        if math.abs(res_py[i].to_float64() - res_simd_mojo[i]) > 1e-6:
            py.print(py.str("Mismatch at index {}").format(i))
            py.print(
                py.str("Python: {}, Mojo SIMD: {}").format(
                    res_py[i].to_float64(), res_simd_mojo[i]
                )
            )
            raise "Test fail"

        # compare mojo with mojo simd now
        if math.abs(res_mojo[i] - res_simd_proper_mojo[i]) > 1e-6:
            py.print(py.str("Mismatch at index {}").format(i))
            py.print(
                py.str("Mojo: {}, Mojo SIMD (proper): {}").format(
                    res_mojo[i], res_simd_proper_mojo[i]
                )
            )
            raise "Test fail"


fn main() raises:
    test()

    var arr = stack_allocation[bench_size, type]()
    # seed(1)
    rand(arr, bench_size)

    var py = Python.import_module("builtins")
    # _ = py.print(py.str("Starting benchmark, size {}...").format(size))

    var dummy = stack_allocation[bench_size, type]()

    var res = softmax[bench_size](arr, dummy)

    @always_inline
    @parameter
    fn worker():
        var bres = softmax[bench_size](arr, dummy)
        benchmark.keep(bres)  # do not optimize out

    var r = benchmark.run[worker](max_runtime_secs=5)
    py.print(py.str("Mean time (native): {}ms").format(r.mean("ms")))

    var _r = softmax_simd_proper[bench_size, _simd_width](arr, dummy)

    @always_inline
    @parameter
    fn worker_simd_proper():
        var bres = softmax_simd_proper[bench_size, _simd_width](
            arr, stack_allocation[bench_size, type]()
        )
        benchmark.keep(bres)  # do not optimize out

    var r_simd = benchmark.run[worker_simd_proper](max_runtime_secs=5)
    py.print(py.str("Mean time (SIMD A): {}ms").format(r_simd.mean("ms")))

    if bench_size == 2048:
        var simd_arr = arr.load[width=bench_size](0)
        var _r = softmax_simd[bench_size](simd_arr)

        @always_inline
        @parameter
        fn worker_simd():
            var bres = softmax_simd[bench_size](simd_arr)
            benchmark.keep(bres)  # do not optimize out

        var r_simd = benchmark.run[worker_simd](max_runtime_secs=5)
        py.print(py.str("Mean time (SIMD B): {}ms").format(r_simd.mean("ms")))
