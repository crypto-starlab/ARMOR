Shared Fig4 implementation

Run ../lan.sh for the complete method sequence. Each method also has ../<method>/lan.sh and lan.py.
run_benchmark_lan.py orchestrates a single method; run_norm_role.sh starts an individual role. norm_benchmark.py contains the method arithmetic. check_native.py checks the active native binding before each run. tools/render_results.py renders every results.csv with the same layout.

Output is ../<method>/results/. Run tags are fixed to results. No timestamp-named result directories are created.
