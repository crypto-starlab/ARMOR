# Python Wrapper for Phantom

Please refer to the [Phantom API Documentation](https://encryptorion-lab.gitbook.io/phantom-fhe/usages/python-applications) for more information.

## Features

- Support common APIs of word-wise scheme BFV, BGV, and CKKS

## Roadmap

- [x] Support word-wise schemes including BGV, BFV
- [ ] Add more examples

## ARMOR optimized constant extraction

`extract_constant_optimize(context, decrypted_ntt)` uses a single multi-block
reduction kernel, reusable buffers per host thread and CUDA device, and mapped
pinned host memory for the tiny output. Non-mapping devices use a pinned D2H
transfer instead. Every call synchronizes its stream before returning a copied
RNS vector; decrypted inputs and context/key pointers are never cached.
The last block completes the final modular reduction using the memory-fence
ordering described in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html#memory-fence-functions).
Completion counters wrap to zero and are initialized only when their buffers grow.

The A100 configuration uses 128 threads per block and
`min(128, max(1, N / 256))` blocks per RNS row (64 for N=16384).
Vectorized pair loads avoid a serial loop for this default shape. Launch sizes,
one-block reduction and two-kernel alternatives were compared using actual
extract-plus-decode measurements; the fastest measured configuration was selected.

`armor_extract_rns_constant`, `armor_ntt_partial_reduce_kernel`,
`armor_ntt_final_reduce_kernel`, original extract/decode functions and
`armor_coeff.py` are unchanged. Gaussian extraction noise remains disabled.
The former optimized implementation is available as `extract_constant_optimize_v1`
for reproducible before/after comparisons. Temporary tuning APIs were removed.

Run from the phantom-fhe directory using the existing build:

```bash
cmake --build build --target pyPhantom -j 8
PYTHONPATH="$PWD/build/lib" LD_LIBRARY_PATH="$PWD/build/lib" \
  python python/examples/armor_coeff_optimize_bench.py --iterations 1000
PYTHONPATH="$PWD/build/lib" LD_LIBRARY_PATH="$PWD/build/lib" \
  python python/examples/armor_extract_optimize_validate.py
```

The benchmark times full coefficient decode for SOTA, and extraction plus the
unchanged constant decoder for OURS, OPT_V1 and OURS_OPT. Encryption, arithmetic,
rescaling and decryption are excluded. Twenty warmups initialize reusable buffers;
first-use allocations and later buffer growth are not represented by steady-state
medians. Timing order rotates to reduce systematic bias.

Measured on NVIDIA A100-SXM4-80GB, N=16384, [62,52,52,52], scale=2^88,
size_P=0, extraction noise OFF, existing build settings, 1000 samples per path:

| Path | Median decode time | Mean decode time |
| --- | ---: | ---: |
| SOTA | 0.067417 ms | 0.070900 ms |
| OURS | 0.038091 ms | 0.039954 ms |
| OPT_V1 | 0.035840 ms | 0.038022 ms |
| OURS_OPT | 0.027849 ms | 0.029397 ms |

The optimized decode path is 22.3% faster than OPT_V1 and 26.9% faster than OURS.
Full output is saved in `examples/armor_coeff_optimize_results.txt`.

Validation checks exact RNS equality, independent norm correctness, repeated
calls, cache growth, non-NTT rejection, N=4096 through 131072, 1/2/3 ciphertext
components, 1/2/3/4 remaining primes, host threads and device switching.
CUDA Memcheck reports zero errors. CUDA Racecheck with
`--quick --no-threads` reports zero hazards. On this CUDA 12.6 setup Racecheck
with the Python host thread pool reported no hazards but failed during process
shutdown; ordinary threaded validation and threaded Memcheck completed normally.

### Additional parameter set: [62,58,58], scale 2^74

From the phantom-fhe directory:

```bash
PYTHONPATH="$PWD/build/lib" LD_LIBRARY_PATH="$PWD/build/lib" \
  python python/examples/armor_coeff_optimize_bench.py \
  --iterations 1000 --chain 62 58 58 --scale-bits 74
```

The parameter overrides affect only this benchmark process; `armor_coeff.py`
is unchanged. Exactly one rescale runs before timing, and checks enforce
chain_index=1 and two extracted residues, i.e. Q1=q0*q1. The constant decoder
uses `2^148/q2` (approximately `2^90`) as its remaining scale. SOTA also decodes
the same level. Encryption, arithmetic, rescale and decryption remain outside
the timer; extraction noise is OFF. All-ones, zeros and random norms pass.

A100-SXM4-80GB, 20 warmups, 1000 samples; full results are saved in
`examples/armor_coeff_optimize_chain62_58_58_scale74_results.txt`:

| Path | Median decode time |
| --- | ---: |
| SOTA | 0.063916 ms |
| OURS | 0.035303 ms |
| OPT_V1 | 0.033147 ms |
| OURS_OPT | 0.026225 ms |

CPU results for this parameter set are in
`../../../../Armor/seal_coeff_cpu/results_cpu_chain62_58_58_scale74.txt`.
SEAL uses C++ and its CPU-specific extraction optimization; Phantom measurements
include Python binding overhead, GPU launch/synchronization and result transfer.
