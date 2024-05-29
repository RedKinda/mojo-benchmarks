import sys
import benchmark
from pathlib import Path


# fn bench(
#     bench_id: StringRef, bench_time: Int = 5
# ) raises -> Dict[StringRef, benchmark.Report]:
#     return Dict[StringRef, benchmark.Report]()


fn save_report(
    bench_id: StringRef, fn_name: StringRef, report: benchmark.Report
) raises:
    var fname = String("").join(fn_name, "_", str(bench_size), "_mojo.json")
    var path = Path(".") / "bench_times" / bench_id / fname
    print(fname, end=" - ")
    print(report.mean("ms"))

    var json = String("{")
    json += '"mean": ' + str(report.mean("ns")) + ","
    json += '"warmup_time": ' + str(report.warmup_duration) + ","
    json += '"bench_time": ' + str(report.duration("s")) + ","
    json += '"file": "' + str(fn_name) + '",'
    json += '"size": ' + str(bench_size) + ","
    # add times in nanoseconds
    json += '"times": ['
    for batch in report.runs:
        var time = batch[].mean("ns")
        json += str(time) + ","

    # remove trailing comma
    json = json[:-1]
    json += "]}"

    with open(path, "w") as f:
        f.write(json)


fn main() raises:
    var bench_id = sys.argv()[1] if len(sys.argv()) > 1 else "mojobench"
    var bench_duration = int(sys.argv()[2]) if len(sys.argv()) > 2 else 5
    var reports = bench(bench_id, bench_duration)
    for pair in reports.items():
        var fn_name = pair[].key
        var report = pair[].value
        save_report(bench_id, fn_name, report)
