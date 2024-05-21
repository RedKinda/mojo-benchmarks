#!/usr/bin/env python3
# small cmd line util to open benches/{name}/{name}.{suffix}
# and run all of the bench functions in it

import sys
import os
import time
from typing import Literal


def shield_exceptions(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return inner


def compile_size(
    name,
    size,
    format: Literal["mojo", "py", "rs"],
):
    to_match = "bench_size = "
    if format == "rs":
        to_match = "bench_size: usize = "

    # open file under /tmp, find line that has `size = ` in it and replace the value with the new size
    with open(f"benches/{name}/{name}.{format}", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if to_match in line and not line[0].isspace():
                words = line.split()
                do_semicolon = format == "rs"
                words[-1] = f"{size}{';' if do_semicolon else ''}"
                lines[i] = " ".join(words) + "\n"

    with open(f"tmp/{name}.{format}", "w") as f:
        f.writelines(lines)


def compile_mojo_benchmain(name):
    # we need to replace the main() function in the mojo file with the contents of bench_suite.mojo
    with open("bench_suite.mojo", "r") as f:
        bench_suite = f.readlines()

    newcontents = []
    with open(f"tmp/{name}.mojo", "r") as f:
        lines = f.readlines()
        gotmain = False
        for i, line in enumerate(lines):
            if gotmain:
                if line.startswith("    "):
                    pass
                else:
                    gotmain = False
                    break

            elif "fn main()" in line:
                gotmain = True
            else:
                newcontents.append(line)

        newcontents.extend(bench_suite)

    with open(f"tmp/{name}.mojo", "w") as f:
        f.writelines(newcontents)


@shield_exceptions
def bench_py(name, size, bench_id, warmup_time=3, bench_time=5):
    print(f"----- Running {name} - python")
    # copy file
    os.system(f"cp benches/{name}/{name}.py tmp/")
    compile_size(name, size, "py")
    os.system(
        f"python3 bench_suite.py tmp.{name} {bench_id} {warmup_time} {bench_time}"
    )


@shield_exceptions
def bench_pypy(name, size, bench_id, warmup_time=1, bench_time=5):
    print(f"----- Running {name} - pypy")
    # copy file
    os.system(f"cp benches/{name}/{name}.py tmp/")
    compile_size(name, size, "py")
    os.system(f"pypy3 bench_suite.py tmp.{name} {bench_id} {warmup_time} {bench_time}")


@shield_exceptions
def bench_mojo(name, size, bench_id, bench_time=5):
    print(f"----- Running {name} - mojo")
    os.system(f"cp benches/{name}/{name}.mojo tmp/")
    compile_size(name, size, "mojo")
    compile_mojo_benchmain(name)
    os.system(f"mojo tmp/{name}.mojo {bench_id} {bench_time}")


@shield_exceptions
def bench_rust(name, size, bench_id, bench_time):
    print(f"----- Running {name} - rust")
    os.system(f"cp benches/{name}/{name}.rs tmp/")
    compile_size(name, size, "rs")
    os.system(
        f"rustc tmp/{name}.rs -o tmp/{name}_rs -C opt-level=3 -C target-cpu=native -C lto -C codegen-units=1 -C panic=abort && tmp/{name}_rs {size} {bench_id} && rm tmp/{name}_rs"
    )


size_defaults = {
    "crc16": 100000,
    "quicksort": 10000,
    "softmax": 2048,
    "matmul": 256,
}


def do_bench(name, size, bench_id, bench_time=5):
    path = os.path.join("benches", name, name + ".py")
    if not os.path.exists(path):
        print("no bench named", name)
        return

    print(f"Benching {name} (size {size})")
    bench_py(name, size, bench_id, bench_time)
    bench_pypy(name, size, bench_id, bench_time)
    bench_mojo(name, size, bench_id, bench_time)
    bench_rust(name, size, bench_id, bench_time)

    print(f"----- Done {name}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    bench_id = sys.argv[2] if len(sys.argv) > 2 else None
    bench_time = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    # mkdir tmp
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    if name == "all":
        # bench all
        print("Benching all")
        bench_id = bench_id or f"all_{str(int(time.time()))}"
        # mkdir under bench_times
        os.mkdir(f"bench_times/{bench_id}")

        for name in os.listdir("benches"):
            if name.startswith("."):
                continue

            do_bench(name, size_defaults[name], bench_id, bench_time)
            print("\n--------------------\n")

        print("Done all!")
        return

    do_bench(name, size_defaults[name], f"{name}_{size_defaults[name]}", bench_time)

    # cleanup tmp
    # os.system("rm -r tmp")


if __name__ == "__main__":
    main()
