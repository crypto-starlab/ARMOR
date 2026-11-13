"""Validate exact extraction, cache growth, input validation and device switching."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyPhantom as phantom
import armor_coeff as example
from armor_coeff_optimize_bench import decrypt_product


def check(context, ciphertext):
    reference = phantom.armor_extract_constant(context, ciphertext)
    phantom.cudaStreamSychronize()
    optimized = phantom.extract_constant_optimize(context, ciphertext)
    assert optimized == reference, (optimized, reference)
    # Returned vectors must remain valid after the reusable workspace is overwritten.
    saved = list(optimized)
    for _ in range(10):
        assert phantom.extract_constant_optimize(context, ciphertext) == saved
    assert optimized == saved
    return len(saved)


def fixture(log_n):
    example.n = 1 << log_n
    context, _ = example.set_context()
    sk, mult_key, _, _, pk, encoder = example.set_key_setting(context, example.size_P)
    values = np.random.default_rng(log_n).uniform(-1, 1, example.n)
    encrypted = example.encrypt_list(context, encoder, pk, values, example.scale, example.size_P)
    conjugated = phantom.armor_rotate_without_key_switching(context, encrypted, 0)
    multiplied = phantom.armor_multiply(context, encrypted, conjugated)
    decrypted, _, product = decrypt_product(context, encoder, pk, sk, mult_key, values)
    return context, sk, mult_key, pk, encoder, encrypted, multiplied, decrypted, product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-threads', action='store_true', help='Disable the host thread pool for GPU Racecheck')
    args = parser.parse_args()
    for log_n in ([14, 16] if args.quick else [12, 13, 14, 15, 16, 17]):
        items = fixture(log_n)
        context, _, _, _, _, encrypted, multiplied, decrypted, _ = items
        lengths = [check(context, ct) for ct in (decrypted, encrypted, multiplied)]
        for chain_index in [1, 2, 3]:
            reduced = phantom.armor_to_intt(context, encrypted)
            reduced = phantom.armor_rescale_to(context, reduced, chain_index)
            reduced = phantom.armor_to_ntt(context, reduced)
            lengths.append(check(context, reduced))
        non_ntt = phantom.armor_to_intt(context, encrypted)
        try:
            phantom.extract_constant_optimize(context, non_ntt)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError('non-NTT input was accepted')
        if not args.no_threads:
            with ThreadPoolExecutor(max_workers=4) as pool:
                assert all(v == lengths[0] for v in pool.map(lambda _: check(context, decrypted), range(8)))
        print(f'PASS N={1<<log_n}: 1/2/3 components, 1/2/3/4 primes, result lengths={lengths}, threads={not args.no_threads}')
    if not args.quick:
        # Use CUDA's public runtime API because pyPhantom has no device-selection binding.
        runtime = ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12')
        runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        def select(device):
            assert runtime.cudaSetDevice(device) == 0
        select(0)
        on_zero = fixture(14)
        select(1)
        on_one = fixture(14)
        for device, items in [(0, on_zero), (1, on_one), (0, on_zero), (1, on_one)]:
            select(device)
            check(items[0], items[7])
        del items
        select(1)
        del on_one
        phantom.cudaStreamSychronize()
        select(0)
        del on_zero
        phantom.cudaStreamSychronize()
        print('PASS device 0 -> 1 -> 0 -> 1: independent workspaces')
    print('All optimized extraction checks passed.')


if __name__ == '__main__':
    main()
