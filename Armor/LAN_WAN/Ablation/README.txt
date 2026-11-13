Fig4: LAN Norm Computation Experiments

RUN ALL METHODS
  bash lan.sh
Default ringdims: 32768, then 65536. For each ringdim: wo_Dist -> wo_Lazy -> wo_Coeff -> wo_Ext -> ARMOR.

RUN ONE METHOD
  bash wo_Dist/lan.sh
  bash wo_Lazy/lan.sh
  bash wo_Coeff/lan.sh
  bash wo_Ext/lan.sh
  bash ARMOR/lan.sh

Each method has lan.sh and lan.py. Each shell command launches all four roles.

SHARED SETTINGS
Server1-Server2 LAN: 3000 Mbit/s, 0.1 ms added one-way delay.
Client-Server1: host loopback, unshaped.
Random input sizes: 62, 184, 367 blocks of 16384 scalars.
One warmup iteration and five measured iterations per input size; seed 2023.
One logical client. No dataset training or model-poisoning attack.

CONSISTENT OUTPUTS UNDER EACH METHOD
  results_<ringdim>/results.csv
  results_<ringdim>/results.png
  results_<ringdim>/results.pdf
  results_<ringdim>/experiment.json
  results_<ringdim>/network_status.log
  results_<ringdim>/preflight.log
  results_<ringdim>/keycenter.jsonl and keycenter.stdout.log
  results_<ringdim>/server1.jsonl and server1.stdout.log
  results_<ringdim>/server2.jsonl and server2.stdout.log
  results_<ringdim>/client.jsonl and client.stdout.log
  results_<ringdim>/blocks_0062/summary.json
  results_<ringdim>/blocks_0184/summary.json
  results_<ringdim>/blocks_0367/summary.json

CSV column definitions and plot layouts are identical across all methods. Time values in the CSV are milliseconds. Completed experiments are skipped on restart. Failed preflight records are preserved under results_<ringdim>/attempts/ before retrying. Existing partial measurements are preserved and are not overwritten. Each method uses a distinct TCP port range.

Preview all commands without running GPU work or changing network settings:
  bash lan.sh --dry-run

RINGDIM SWEEP
  bash lan.sh --dry-run
  bash lan.sh
Optional baseline: RINGDIMS="16384 32768 65536" bash lan.sh
One method: bash wo_Lazy/lan.sh --ringdim 65536

Original input lengths stay 62/184/367 * 16384 scalars regardless of ringdim.
Final ciphertext chunks are zero padded; original data is never dropped.
Coefficient methods (ARMOR, wo_Dist, wo_Lazy, wo_Ext):
  ringdim 32768: 31, 92, 184 chunks; padding 0, 0, 16384 scalars.
  ringdim 65536: 16, 46, 92 chunks; padding 32768, 0, 16384 scalars.
wo_Coeff packs ringdim/2 slots:
  ringdim 32768: 62, 184, 367 chunks; no padding.
  ringdim 65536: 31, 92, 184 chunks; padding 0, 0, 16384 scalars.
CSV records ringdim, ciphertext_chunks and padding_elements.
Summary JSON also records chunk_width and padding_elements.
Total: 2 ringdims * 5 methods * 3 inputs = 30 cases;
1 warmup + 5 measured iterations each = 180 iterations.
Completed dimension/method combinations are skipped on rerun.
Default timeout: 14400 seconds per method/ringdim (override --timeout).
Use tmux for a long sweep and authenticate sudo before starting.
