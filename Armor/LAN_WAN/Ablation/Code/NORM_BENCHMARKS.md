# Four-process norm benchmark: wo_Dist, wo_Lazy, M3, wo_Coeff, Yasuda, ARMOR

This benchmarks the **norm stage only**, using exactly the three random input
lengths below. It never loads a model or dataset and does not run federated training,
attacks, similarity filtering, aggregation or layer sampling. Each method has
its own `keycenter.py`, `server1.py`, `server2.py`, `client.py` in its named folder.
The entry files share `norm_benchmark.py` for transport, timing and common work;
the arithmetic dispatch remains method-specific.

The root-level `client.py`, `server1.py`, `server2.py` are ARMOR FL roles again:
the Yasuda branches were removed. The root FL KeyCenter rejects a different
norm protocol, directing comparisons to these folders. Single-process encrypted
`main.py` still supports its prior Yasuda mode independently.

## Random input sizes

The default four-process run sweeps all three cases, with warmup and measured
iterations applied independently to each size. N is 16384 by default.

| `--input_blocks` | Random scalar inputs | Coefficient chunks | wo_Coeff slot chunks |
|---:|---:|---:|---:|
| 62 | 62 × 16384 = 1,015,808 | 62 | 124 |
| 184 | 184 × 16384 = 3,014,656 | 184 | 368 |
| 367 | 367 × 16384 = 6,012,928 | 367 | 734 |

Input lengths always use **16384** as the block size, independently of ringdim.
Values are float64 samples from N(0, 0.01²), produced using NumPy's local seeded
random generator (default seed 2023). A fixed seed and input size produce the
same vector for all six methods. SHA-256 digests are printed and stored in each
size's summary/CSV so comparisons can verify matching inputs. Yasuda uploads
both coefficient ciphertext lists, while slot packing uses N/2 values per
ciphertext: comparisons use identical input counts, not identical ciphertext counts.
`--model` and `--elements` have been removed.

## Algorithms and parameters

| Folder | Packing | Server1 HE work | Server1/Server2 decryption schedule |
|---|---|---|---|
| wo_Dist | N coefficients | Key-switched conjugation + general multiplication per chunk; add products; one relinearization; rescale | One partial RPC per vector; constant output |
| wo_Lazy | N coefficients | No-key-switch automorphism + specialized ARMOR product, rescale each chunk | One partial RPC **per chunk**; add decoded squared norms |
| wo_Ext | N coefficients | Same as ARMOR: no-key-switch automorphism + specialized products; add products; rescale once | One RPC; full partial polynomial returned; S1 combines polynomials and coefficient-decodes coefficient zero |
| wo_Coeff | N/2 real slots | Square each chunk; add products; relinearize/rescale once; hoist rotations 1..15; add power-of-two rotations from 16 | One RPC; full coefficient partial ciphertext returned; combine and CKKS slot decode |
| Yasuda | N coefficients, two encryptions | Client sends independent encryptions of f(X), f(X^-1); S1 general-multiplies pairs; adds products; rescale once | One RPC; additive s and s² shares; full partial polynomial returned; S1 combines polynomials then extracts/decodes constant |
| ARMOR | N coefficients | No-key-switch automorphism + specialized ARMOR products; add products; rescale once | One RPC; additive s and s·sigma(s) shares; constant output |

Shared FL modules, `phantom-fhe`, and network configuration remain in the parent directory.

Reference algorithms: `wo_Dist.py`, `wo_Lazy.py`, `M3.py`, `Ours.py`. The slot reference was
found at `ARMOR copy/Armor/wo_Coeff_Dynamic_Hoisting.py` (the root reference file
is absent). These scripts execute GPU work on import, so the benchmark extracts
the algorithms rather than importing the scripts. wo_Ext follows the revised full-polynomial ARMOR variant described above, rather than the legacy M3.py algorithm. It fixes reference timing
issues (GPU synchronization, early returns and iteration-counter resets) and
uses **all** chunks instead of their old random subset loop.

The reference scripts use a single secret key. Here decryption is adapted to
two additive secret-key shares to measure actual Server1↔Server2 communication;
this is not a claim that the original scripts defined a network protocol.
wo_Dist/wo_Lazy/ARMOR return extracted constant partials. wo_Ext uses ARMOR arithmetic and additive s and s·sigma(s) shares, but both servers flood all coefficients and keep full polynomial partials. Server2 serializes its full partial; Server1 combines the two polynomials and coefficient-decodes coefficient zero. Yasuda returns the full
partial polynomial in a serialized ciphertext container; Server1 adds both
partial polynomials before extracting and decoding the constant. Its response
byte count includes the complete serialized polynomial, not just the constant.
Both Yasuda servers add independent Karney DGS noise (standard deviation 2^52
in the current default) to all N coefficients of their own partial polynomial.
Noise is added after INTT, before Server2 serializes its response. Server1
combines the coefficient-form partials and coefficient-decodes the result,
selecting coefficient zero without adding another constant-only noise sample.
Older Yasuda runs with constant-only responses must be rerun with a fresh tag.
The slot method returns a full partial ciphertext and performs standard CKKS
slot decoding. No master
secret key is distributed. Only Server1 receives rotation/relinearization keys.

Defaults follow the supplied references: N=16384, scale=2^88. wo_Dist/Slot use
[62,52,52,52,60], size_P=1; ARMOR/wo_Lazy/wo_Ext/Yasuda use [62,52,52,52], size_P=0.
`summary.json` records the exact parameters; the methods do not have identical
modulus chains/packing capacities. Use identical seed, random input sizes and
network profile across methods. Logical clients use the same random vector and
run sequentially; no data partitioning, model loading or local training occurs.


## Build

Each role requires CUDA and the same matching binding. New evaluation-key
serialization is required for wo_Dist/Slot. The share and public/ciphertext
serialization are also used by the existing implementation.

```bash
conda activate ARMOR
cd ~/ARMOR/ARMOR_copy/Armor/Figure4
cmake --build ../phantom-fhe/build --target pyPhantom -j 4
```

## LAN run: one method, four terminals

The existing LAN profile is 3 Gbit/s per direction and added RTT 0.2 ms.
It shapes only the host↔Server2 link. Client uploads stay on the host network;
Client↔Server1 byte counts are measured, but their timings are not a simulated
remote client link. To create or switch the profile, on the host:

```bash
bash ../network_profile.sh setup lan       # first creation
# bash ../network_profile.sh profile lan   # existing namespace
bash ../network_profile.sh status
```

In all four terminals:

```bash
conda activate ARMOR
cd ~/ARMOR/ARMOR_copy/Armor/Figure4
export ARMOR_AUTHKEY='same-experiment-key-in-all-four-terminals'
export RUN_TAG=lan_rep1
```

Run one line per terminal, in order. Wait for the listening message at each step.
One invocation measures all three requested random input sizes:

```bash
# Terminal 1: all authoritative benchmark settings are supplied here
bash run_norm_role.sh ARMOR keycenter --profile lan \
  --input_blocks 62 184 367 --iterations 5 --warmup 1 --num_clients 1

# Terminal 2: enters armor-s2 automatically, via sudo
bash run_norm_role.sh ARMOR server2 --profile lan

# Terminal 3
bash run_norm_role.sh ARMOR server1 --profile lan

# Terminal 4
bash run_norm_role.sh ARMOR client --profile lan
```

KeyCenter exits after distributing keys. Server1 and Server2 run only the norm
protocol. Each input is freshly encrypted for every iteration/client. Each
result is checked against the plaintext sum of squares; measured failures cause
Server1 to exit with an error after writing the per-size summaries. The expected norm
is sent solely for benchmark verification; it is not a privacy-protocol input.

To run only one size, set KeyCenter's `--input_blocks 62` (or `184`, `367`).
Omitting `--input_blocks` defaults to all three. For a short pipeline check use
`--iterations 1 --warmup 0`; input sizes remain exact. For more repetitions use
`--iterations 50`. Each role has a local `--gpu` override. Model options are not
supported by this random-input benchmark.

After all roles finish, repeat the same four commands with `ARMOR` replaced by
`wo_Dist`, `wo_Lazy`, `M3`, `wo_Coeff`, `Yasuda`. Use identical settings on KeyCenter.
Methods use separate output directories and can share `RUN_TAG=lan_rep1`.
For repeated runs of the same method set a fresh tag in **all terminals**, e.g.
`lan_rep2`. Do not reuse a tag: JSONL logs append, which would mix setup traffic.
Run methods sequentially, not concurrently on the same ports/GPU.

## WAN or no shaping

After finishing/stopping all roles:

```bash
bash ../network_profile.sh profile wan
bash ../network_profile.sh status
```

Set `RUN_TAG=wan_rep1` in all terminals and use `--profile wan` on every command.
The current WAN profile is 100 Mbit/s per direction, added RTT 80 ms. For
loopback-only validation use `--profile local` with no namespace/shaping.
Selecting a launcher profile does **not** change tc settings automatically.
Actual ping/iperf measurements should be saved with each experiment; configured
rates and delays are not measured performance. The existing network guide has
the bidirectional iperf commands. Use private/VPN networks for this trusted
pickle/key transport.

## Output and interpretation

```text
runs/norm_benchmarks/ARMOR/lan_rep1/
  keycenter.jsonl
  server1.jsonl
  server2.jsonl
  client.jsonl
  blocks_0062/summary.json
  blocks_0184/summary.json
  blocks_0367/summary.json
```

For each measured norm, `server1.jsonl` includes:

- `client_upload_bytes`: actual serialized Client→Server1 input frame size.
  Yasuda includes **both** ciphertext lists; Slot has N/2 packing capacity.
- `s1_s2_tx_bytes`, `s1_s2_rx_bytes`: actual request/response frame sizes.
  Count these once from Server1; do not add mirrored Server2 traffic again.
- `client_encrypt_s`: synchronized encode/encrypt time, excluding serialization.
- `he_s`: synchronized Server1 HE time, including rescale and slot reduction.
- `rpc_s`: Server1 measured partial-decryption RPC elapsed time, including
  serialization, network, waiting, and Server2 computation.
- `local_decrypt_decode_s`: synchronized Server1 partial/decode time.
- `server2_compute_s`: Server2's own synchronized partial-decryption time.
- `norm_total_s`: Server1 wall time for all HE + partial RPCs + local decoding.
  It excludes client encryption/upload and key setup. **Do not add** this to its
  component times or add Server2 compute again to RPC.
- `jobs`: 1 for wo_Dist/wo_Ext/Slot/Yasuda/ARMOR; chunk count for wo_Lazy.
- `norm_squared`, `norm`, `expected_norm_squared`, `absolute_error`, `valid`.

Each per-size `summary.json` contains mean/std/min/max/total for measured iterations only;
`warmup` iterations are excluded consistently. It records input length, packing,
modulus chain and number of clients through the configuration/results logs.
The same keys are reused across the three sizes, and setup bytes in the CSV
refer to the shared whole-run setup; do not add them once per size. The summary
tool reads only per-size summaries, excluding old model-vector results.
Client `upload` records measure send→receipt acknowledgement latency separately
from later norm computation. Bytes include serialized application framing but
exclude authentication, TCP/IP headers, retransmissions and network-layer wire
cost. Logs never fabricate communication from ciphertext count estimates.

Compare all finished runs:

```bash
python summarize_norm_benchmarks.py
python summarize_norm_benchmarks.py --csv results/norm_comparison.csv
```

The CSV additionally includes separate KeyCenter setup send/receive bytes and
key-generation time. Setup is not counted as per-norm traffic. Default tolerances
are rtol=atol=1e-3; these can be set on KeyCenter. Preserve numerical errors and
survivor-free norm checks when reporting latency, rather than reporting timings
for incorrect results.

## Validation

```bash
PYTHONPATH=.. python -m unittest test_norm_benchmark test_distributed test_partial_modes -v
```

CPU arithmetic tests verify all methods' full-vector norms and operation counts;
CUDA-gated tests verify actual ciphertexts, serialized evaluation keys, two-party
partials and decoding for all six methods. The restricted development session
has no working GPU or network-administration rights: native compilation and CPU
checks can be completed here, but real CUDA/LAN/WAN measurements must run on the
experiment host. Do not interpret CPU test durations as benchmark timings.
