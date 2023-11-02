using PkgBenchmark

target = "HEAD"
baseline = "main"

bench_target = benchmarkpkg("BeliefPropagation", target, retune=true)
bench_base = benchmarkpkg("BeliefPropagation", baseline, retune=true)
comparison = judge(bench_target, bench_base)

export_markdown(stdout, comparison; export_invariants=true)