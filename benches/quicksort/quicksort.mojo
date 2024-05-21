import sys
import time
from random import rand, seed
from python import Python
import memory
import benchmark

alias type = DType.int8
alias bench_size = 1000000


fn quicksort(inout data: DTypePointer[type], left: Int, right: Int):
    if left >= right:
        return

    var pivot = data[right]
    var i = left - 1

    for j in range(left, right):
        if data[j] <= pivot:
            i = i + 1
            var tmp = data[i]
            data[i] = data[j]
            data[j] = tmp

    var tmp = data[i + 1]
    data[i + 1] = data[right]
    data[right] = tmp

    i += 1

    quicksort(data, left, i - 1)
    quicksort(data, i + 1, right)


def test():
    alias testlen = 100
    var data = stack_allocation[testlen, type]()
    rand(data, testlen)

    quicksort(data, 0, testlen - 1)

    for i in range(testlen - 1):
        if data[i] > data[i + 1]:
            raise "Sort failed"


fn initialize() -> DTypePointer[type]:
    var arr = DTypePointer[type].alloc(bench_size)
    rand(arr, bench_size)
    return arr


fn bench(
    bench_id: StringRef, bench_time: Int = 5
) raises -> Dict[StringRef, benchmark.Report]:
    test()
    var arr = initialize()

    @always_inline
    @parameter
    fn worker():
        var temp = stack_allocation[bench_size, type]()
        memcpy[bench_size](temp, arr)
        quicksort(temp, 0, bench_size - 1)

    var r = benchmark.run[worker](
        min_runtime_secs=bench_time, max_runtime_secs=bench_time
    )

    var res = Dict[StringRef, benchmark.Report]()
    res["quicksort"] = r
    return res


fn main() raises:
    test()
    var arr = initialize()
    var py = Python.import_module("builtins")

    var reports = bench("quicksort", 5)
    var r = reports["quicksort"]

    py.print(py.str("Mean time: {}ms").format(r.mean("ms")))
