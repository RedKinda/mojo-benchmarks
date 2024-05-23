import os
import signal
import sys
import time
import json


def save_bench_results(bench_id, name, times, **kwargs):
    print(f"Saving {name} times - mean: {(sum(times) / len(times))/1000/1000}ms")
    suffix = "py"
    if "pypy" in sys.version.lower():
        suffix = "pypy"

    fname = f"bench_times/{bench_id}/{name}_{suffix}.json"
    to_save = {"mean": sum(times) / len(times), **kwargs, "times": times}
    with open(fname, "w") as f:
        json.dump(to_save, f)


def signal_handler(sig, frame):
    python_slow = False
    try:
        while frame:
            if "_SPECIAL_WARMUP_VAR" in frame.f_locals:
                python_slow = True
                break
            frame = frame.f_back

    except Exception as e:
        print(e)

    if python_slow:
        raise RuntimeError("warmup took longer than bench time! skipping...")


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

    bench_time_ns = bench_time * 1000 * 1000 * 1000

    test()

    for fname in functions:
        f = getattr(b, fname)

        def warmup():
            # warmup
            _SPECIAL_WARMUP_VAR = "warmup"
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(bench_time)
            start = time.time()
            while time.time() - start < warmup_time:
                f(input_param)

            signal.alarm(0)  # disable the alarm

        try:
            warmup()
        except RuntimeError as e:
            print(e)
            continue

        start = time.perf_counter_ns()
        times = [start]
        while time.perf_counter_ns() - start < bench_time_ns:
            f(input_param)
            times.append(time.perf_counter_ns())

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
