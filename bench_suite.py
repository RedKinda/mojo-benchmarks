import os
import sys
import time
import json


def save_bench_results(bench_id, name, times, **kwargs):
    print(f"Saving {name} times - mean: {(sum(times) / len(times))*1000}ms")
    fname = f"bench_times/{bench_id}/{name}_py.txt"
    to_save = {"mean": sum(times) / len(times), **kwargs, "times": times}
    with open(fname, "w") as f:
        json.dump(to_save, f)


def bench(
    file_location,
    *,
    bench_id,
    warmup_time=1,
    bench_time=5,
):
    # import the module from file
    b = getattr(__import__(file_location), file_location.rsplit(".", 1)[1])
    # list all functions in workload starting with bench_
    functions = [
        f for f in dir(b) if f.startswith("bench_") and callable(getattr(b, f))
    ]
    # get the test function
    test = getattr(b, "test")
    input_param = getattr(b, "initialize")()

    test()

    for fname in functions:
        f = getattr(b, fname)
        # warmup
        start = time.time()
        while time.time() - start < warmup_time:
            f(input_param)

        start = time.time()
        times = [start]
        while time.time() - start < bench_time:
            f(input_param)
            times.append(time.time())

        diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
        save_bench_results(
            bench_id,
            fname.removeprefix("bench_"),
            diffs,
            warmup_time=warmup_time,
            bench_time=bench_time,
            file=file_location,
        )


if __name__ == "__main__":
    file_location = sys.argv[1] if len(sys.argv) > 1 else "tmp.crc16"
    bench_id = sys.argv[2] if len(sys.argv) > 2 else str(int(time.time()))
    warmup_time = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    bench_time = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    # mkdir
    if not os.path.exists(f"bench_times/{bench_id}"):
        os.mkdir(f"bench_times/{bench_id}")

    bench(
        file_location=file_location,
        bench_id=bench_id,
        warmup_time=warmup_time,
        bench_time=bench_time,
    )
