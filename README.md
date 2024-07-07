# Benchmarking Mojo

*Note: this project is part of Bachelor's thesis of Red Kalab in 2024. Full work can be found [here](https://www.overleaf.com/read/yyzgdkcqhxhn#88d801).*

## Instructions for running

Build the Dockerfile with command such as `docker build . -t mojo-bench`. Image will take ~5GB on disk. 
To run full benchmarks, with each alg+bench taking ~5 minutes run `docker run -v ./bench_times:/app/bench_times mojo-bench ./_bench.py full run_name 300`. 
This configuration would take approximately 10 hours in total. Resulting benchmark timings will be saved in `./bench_times/run_name`. 
These results can be copied into `./results/data` and evaluated with the notebook in the `./results` folder.

## Customization

The bench utility takes 3 parameters, benchmarks to run, run name and time to run. Valid options for benchmarks to run are `full`, `all`, `softmax`, `matmul`, `crc16` and `quicksort`.
- `full` - runs all benchmarks on all defined sizes
- `all` - runs all benchmarks on default size
- `softmax`/`matmul`/`crc16`/`quicksort` - runs given algorithm on default size

Run name determines the folder name of where the results are stored. Time to run signifies time to run for each algorithm+size combination. 
The `_bench.py` script contains the default/full sizes for benchmarking sizes, and is fully customizable.
