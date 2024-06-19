import benchmark
import sys
from random import rand, seed
from utils.loop import unroll
from python import Python
import math
from algorithm import vectorize
alias type = DType.float64
alias bench_size = 8192
alias _simd_width = 16

alias raw_simd_size = bench_size if bench_size < 8192 else 2

"""

def softmax_native(x: list[float]) -> list[float]:
    maxes = max(x)
    x_exp = [math.exp(xi - maxes) for xi in x]
    x_exp_sum = sum(x_exp)
    probs = [xi / x_exp_sum for xi in x_exp]
    return probs
"""


# fn vectorize[
#     func: fn[Int] (Int) capturing -> None,
#     simd_width: Int,
#     /,
#     *,
#     size: Int,
#     do_unroll: Bool = True,
# ]():
#     alias big_loop_count = size // simd_width
#     alias remainder = size % simd_width
#     alias remainder_offset = size - remainder

#     @parameter
#     fn f_big[i: Int]() capturing:
#         func[simd_width](i * simd_width)

#     @parameter
#     fn f_small[i: Int]() capturing:
#         func[1](remainder_offset + i)

#     if do_unroll:
#         unroll[f_big, big_loop_count]()
#         unroll[f_small, remainder]()
#     else:
#         for i in range(big_loop_count):
#             func[simd_width](i)

#         for i in range(remainder):
#             func[1](remainder_offset + i)


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
        print("called with", simd_width, i)

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

    alias testsize = 18

    var mojoin = stack_allocation[testsize, type]()
    var res_mojo = stack_allocation[testsize, type]()

    pyin = py.list()
    for i in range(testsize):
        var val = pyrandom.random()
        pyin.append(val)
        mojoin[i] = val.to_float64()

    var res_py = pysoftmax.bench_softmax_native(pyin)
    var _a = softmax[testsize](mojoin, res_mojo)
    # var res_simd_mojo = softmax_simd[testsize](mojoin.load[width=testsize](0))
    var res_simd_proper_mojo = softmax_simd_proper[testsize, 4](
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

        # if math.abs(res_py[i].to_float64() - res_simd_mojo[i]) > 1e-6:
        #     py.print(py.str("Mismatch at index {}").format(i))
        #     py.print(
        #         py.str("Python: {}, Mojo SIMD: {}").format(
        #             res_py[i].to_float64(), res_simd_mojo[i]
        #         )
        #     )
        #     raise "Test fail"

        # compare mojo with mojo simd now
        if math.abs(res_mojo[i] - res_simd_proper_mojo[i]) > 1e-6:
            py.print(py.str("Mismatch at index {}").format(i))
            py.print(
                py.str("Mojo: {}, Mojo SIMD (proper): {}").format(
                    res_mojo[i], res_simd_proper_mojo[i]
                )
            )
            raise "Test fail"


fn bench(
    bench_id: StringRef, bench_time: Int = 5
) raises -> Dict[StringRef, benchmark.Report]:
    test()

    var arr = stack_allocation[bench_size, type]()
    for i in range(bench_size):
        arr[i] = random.random_float64()

    var reports = Dict[StringRef, benchmark.Report]()
    var dummy = stack_allocation[bench_size, type]()

    var res = softmax[bench_size](arr, dummy)

    @always_inline
    @parameter
    fn worker():
        var bres = softmax[bench_size](arr, dummy)
        benchmark.keep(bres)  # do not optimize out

    var r = benchmark.run[worker](
        min_runtime_secs=bench_time * 0.75, max_runtime_secs=bench_time
    )
    reports["softmax"] = r

    @always_inline
    @parameter
    fn worker_simd_proper():
        var bres = softmax_simd_proper[bench_size, _simd_width](
            arr, stack_allocation[bench_size, type]()
        )
        benchmark.keep(bres)  # do not optimize out

    var r_simd = benchmark.run[worker_simd_proper](
        min_runtime_secs=bench_time * 0.75, max_runtime_secs=bench_time
    )
    reports["softmax_simd_proper"] = r_simd

    # dirty SIMD benchmark - only do sizes of power two and less than some constant where it seems to crash
    if math.log2[DType.float64, 1](bench_size) % 1 == 0 and bench_size < 8192:
        var simd_arr = arr.load[width=raw_simd_size](0)

        @always_inline
        @parameter
        fn worker_simd():
            var bres = softmax_simd[raw_simd_size](simd_arr)
            benchmark.keep(bres)  # do not optimize out

        var r_simd = benchmark.run[worker_simd](
            min_runtime_secs=bench_time * 0.75, max_runtime_secs=bench_time
        )
        reports["softmax_simd_raw"] = r_simd

    return reports


fn main() raises:
    test()

    var py = Python.import_module("builtins")
    var reports = bench("softmax", 5)

    for pair in reports.items():
        var fn_name = pair[].key
        var report = pair[].value
        py.print(
            py.str("Mean time for {} (ms): {}").format(
                fn_name, report.mean("ms")
            )
        )
