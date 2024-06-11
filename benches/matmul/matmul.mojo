import benchmark
import sys
from random import rand, seed
from python import Python
import math
from algorithm import vectorize, parallelize

alias type = DType.float64
alias bench_size = 1024
alias _simd_width = 16


fn matmul[
    size_ax: Int,
    size_ay: Int,
    # size_bx: Int, - not needed, same as size_ay
    size_by: Int,
](a: DTypePointer[type], b: DTypePointer[type], res_ptr: DTypePointer[type]):
    for i in range(size_ax):
        for j in range(size_by):
            var sum = 0.0
            for k in range(size_ay):
                sum += a[i * size_ay + k] * b[k * size_by + j]

            res_ptr[i * size_by + j] = sum


fn matmul_simd[
    simd_width: Int,
    size_ax: Int,
    size_ay: Int,
    # size_bx: Int, - not needed, same as size_ay
    size_by: Int,
](a: DTypePointer[type], b: DTypePointer[type], res_ptr: DTypePointer[type]):
    for i in range(size_ax):
        for j in range(size_by):

            @parameter
            fn do_sum[simd_width: Int](n: Int):
                res_ptr.store[width=simd_width](
                    i * size_by + n,
                    res_ptr.load[width=simd_width](i * size_by + n)
                    + a[i * size_ax + j]
                    * b.load[width=simd_width](j * size_ay + n),
                )

            vectorize[do_sum, simd_width, size=size_by]()


fn matmul_simd_raw[
    size_ax: Int,
    # size_ay: Int, - for raw we need to assume the same size for both ax and ay
    # size_bx: Int, - not needed, same as size_ax
    size_by: Int,
](a: DTypePointer[type], b: DTypePointer[type], res_ptr: DTypePointer[type]):
    for i in range(size_ax):
        for j in range(size_by):
            res_ptr.store[width=size_ax](
                i * size_by,
                res_ptr.load[width=size_ax](i * size_by)
                + a[i * size_ax + j] * b.load[width=size_ax](j * size_ax),
            )


fn matmul_simd_parallel[
    simd_width: Int,
    size_ax: Int,
    size_ay: Int,
    # size_bx: Int, - not needed, same as size_ay
    size_by: Int,
](a: DTypePointer[type], b: DTypePointer[type], res_ptr: DTypePointer[type]):
    @parameter
    fn row(i: Int):
        for j in range(size_by):

            @parameter
            fn do_sum[simd_width: Int](n: Int):
                res_ptr.store[width=simd_width](
                    i * size_by + n,
                    res_ptr.load[width=simd_width](i * size_by + n)
                    + a[i * size_ax + j]
                    * b.load[width=simd_width](j * size_ay + n),
                )

            vectorize[do_sum, simd_width, size=size_by]()

    parallelize[row](
        size_ax, size_ax
    )  # instead of a forloop over size_ax we parallelize


fn test() raises:
    var x = stack_allocation[4, type]()
    var y = stack_allocation[4, type]()
    var res = stack_allocation[4, type]()
    var res_simd = stack_allocation[4, type]()
    var res_simd_raw = stack_allocation[4, type]()
    var res_simd_parallel = stack_allocation[4, type]()

    # zero out all results
    for i in range(4):
        res[i] = 0
        res_simd[i] = 0
        res_simd_raw[i] = 0
        res_simd_parallel[i] = 0

    x[0] = 1
    x[1] = 2
    x[2] = 3
    x[3] = 4

    y[0] = 5
    y[1] = 6
    y[2] = 7
    y[3] = 8

    var _r = matmul[2, 2, 2](x, y, res)
    var _r_simd = matmul_simd[2, 2, 2, 2](x, y, res_simd)
    var _r_simd_raw = matmul_simd_raw[2, 2](x, y, res_simd_raw)
    var _r_simd_parallel = matmul_simd_parallel[2, 2, 2, 2](
        x, y, res_simd_parallel
    )

    # assert its [[19, 22], [43, 50]]
    if res[0] != 19 or res[1] != 22 or res[2] != 43 or res[3] != 50:
        print(res[0], res[1], res[2], res[3])
        raise "Test failed (native)"

    if (
        res_simd[0] != 19
        or res_simd[1] != 22
        or res_simd[2] != 43
        or res_simd[3] != 50
    ):
        print(res_simd[0], res_simd[1], res_simd[2], res_simd[3])
        raise "Test failed (SIMD)"

    if (
        res_simd_raw[0] != 19
        or res_simd_raw[1] != 22
        or res_simd_raw[2] != 43
        or res_simd_raw[3] != 50
    ):
        print(
            res_simd_raw[0], res_simd_raw[1], res_simd_raw[2], res_simd_raw[3]
        )
        raise "Test failed (raw SIMD)"

    if (
        res_simd_parallel[0] != 19
        or res_simd_parallel[1] != 22
        or res_simd_parallel[2] != 43
        or res_simd_parallel[3] != 50
    ):
        print(
            res_simd_parallel[0],
            res_simd_parallel[1],
            res_simd_parallel[2],
            res_simd_parallel[3],
        )
        raise "Test failed (SIMD+Parallel)"




fn bench(
    bench_id: StringRef, bench_time: Int = 5
) raises -> Dict[StringRef, benchmark.Report]:
    test()

    var inp_a = DTypePointer[type].alloc(bench_size * bench_size)
    var inp_b = DTypePointer[type].alloc(bench_size * bench_size)
    # seed(1)
    # randomize the inputs
    for i in range(bench_size * bench_size):
        inp_a[i] = random.random_float64()
        inp_b[i] = random.random_float64()

    var reports = Dict[StringRef, benchmark.Report]()

    var dummy = DTypePointer[type].alloc(bench_size * bench_size)

    @always_inline
    @parameter
    fn worker():
        var bres = matmul[bench_size, bench_size, bench_size](
            inp_a, inp_b, dummy
        )
        benchmark.keep(bres)  # do not optimize out

    var r = benchmark.run[worker](
        min_runtime_secs=bench_time * 0.75, max_runtime_secs=bench_time
    )
    reports["matmul"] = r

    @always_inline
    @parameter
    fn worker_simd():
        var bres = matmul_simd[_simd_width, bench_size, bench_size, bench_size](
            inp_a, inp_b, dummy
        )
        benchmark.keep(bres)  # do not optimize out

    var r_simd = benchmark.run[worker_simd](
        min_runtime_secs=bench_time * 0.75, max_runtime_secs=bench_time
    )
    reports["matmul_simd"] = r_simd

    @always_inline
    @parameter
    fn worker_simd_raw():
        var bres = matmul_simd_raw[bench_size, bench_size](inp_a, inp_b, dummy)
        benchmark.keep(bres)  # do not optimize out

    var r_simd_raw = benchmark.run[worker_simd_raw](
        min_runtime_secs=bench_time * 0.75, max_runtime_secs=bench_time
    )
    reports["matmul_simd_raw"] = r_simd_raw

    @always_inline
    @parameter
    fn worker_simd_parallel():
        var bres = matmul_simd_parallel[
            _simd_width, bench_size, bench_size, bench_size
        ](inp_a, inp_b, dummy)
        benchmark.keep(bres)  # do not optimize out

    var r_simd_parallel = benchmark.run[worker_simd_parallel](
        min_runtime_secs=bench_time * 0.75, max_runtime_secs=bench_time
    )
    reports["matmul_simd_parallel"] = r_simd_parallel

    # free memory
    inp_a.free()
    inp_b.free()
    dummy.free()

    return reports


fn main() raises:
    test()

    var reports = bench("matmul", 5)

    var py = Python.import_module("builtins")

    for pair in reports.items():
        var fn_name = pair[].key
        var report = pair[].value
        py.print(
            py.str("Mean time for {} (ms): {}").format(
                fn_name, report.mean("ms")
            )
        )
