## Run the benchmarks

```julia
using PkgBenchmark
res = benchmarkpkg("BeliefPropagation")
export_markdown(stdout, res)
```

## Compare across commits
Run `benchmark/run_benchmarks.jl` after (optionally) modifying the `baseline` and `target` variables therein.