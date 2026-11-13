"""Compare original, previous optimized and current optimized decode paths using the timing boundaries of armor_coeff.py."""
import argparse
import statistics
import time

import numpy as np
import pyPhantom as phantom
import armor_coeff as example


def decrypt_product(context, encoder, pk, sk, mult_key, values):
    ciphertext = example.encrypt_list(context, encoder, pk, values, example.scale, example.size_P)
    conjugated = phantom.armor_rotate_without_key_switching(context, ciphertext, example.galois_steps[0])
    product = phantom.armor_multiply(context, ciphertext, conjugated)
    product = phantom.armor_to_intt(context, product)
    product = phantom.armor_rescale_to(context, product, len(example.modulus_chain) - 2)
    product = phantom.armor_to_ntt(context, product)
    decrypted = sk.armor_decrypt_cipher(context, product, mult_key, example.galois_steps[0])
    phantom.cudaStreamSychronize()
    plaintext = sk.armor_decrypt(context, product, mult_key, example.galois_steps[0])
    phantom.cudaStreamSychronize()
    return decrypted, plaintext, product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--chain', type=int, nargs='+', default=list(example.modulus_chain))
    parser.add_argument('--scale-bits', type=int, default=88)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if len(args.chain) < 2:
        parser.error('--chain requires at least two primes')
    if not 0 < args.scale_bits < 126:
        parser.error('--scale-bits must be between 1 and 125')
    example.modulus_chain = args.chain
    example.scale = 2.0 ** args.scale_bits
    print(f'pyPhantom: {phantom.__file__}')
    print(f'N={example.n}, chain={example.modulus_chain}, scale=2^{args.scale_bits}, size_P=0, extraction noise=OFF')
    context, _ = example.set_context()
    sk, mult_key, _, _, pk, encoder = example.set_key_setting(context, example.size_P)
    functions = [phantom.armor_extract_constant, phantom.extract_constant_optimize_v1, phantom.extract_constant_optimize]
    rng = np.random.default_rng(42)
    cases = [('ones', np.ones(example.n)), ('zeros', np.zeros(example.n)),
             ('random', rng.uniform(-1, 1, example.n))]
    for name, values in cases:
        decrypted, plaintext, product = decrypt_product(context, encoder, pk, sk, mult_key, values)
        assert phantom.get_chain_index(product) == len(example.modulus_chain) - 2
        expected = float(np.dot(values, values))
        for function in functions:
            constants = function(context, decrypted)
            phantom.cudaStreamSychronize()
            assert len(constants) == 2, f'Expected Q1=q0*q1, got {len(constants)} residues'
            decoded = phantom.armor_decode_constant_single(context, constants, example.scale, 1.0)
            if not np.isclose(decoded, expected, rtol=1e-9, atol=1e-6):
                raise AssertionError(f'{name}: {function.__name__}: {decoded} != {expected}')
            print(f'{name:6s} {function.__name__:26s}: {decoded:.12f}, abs_error={abs(decoded-expected):.3e}')
    # Use the same all-ones input and decode boundaries as armor_coeff.py.
    # Encryption, multiplication, rescaling and decryption are outside timing.
    decrypted, plaintext, product = decrypt_product(
        context, encoder, pk, sk, mult_key, np.ones(example.n))

    assert phantom.get_chain_index(product) == len(example.modulus_chain) - 2
    print(f'Decode level: Q1=q0*q1, chain_index={phantom.get_chain_index(product)}, remaining primes=2')

    def decode_constant(extract):
        constants = extract(context, decrypted)
        return phantom.armor_decode_constant_single(context, constants, example.scale, 1.0)

    paths = [
        ('SOTA', lambda: encoder.armor_coefficient_decode_and_mult(context, plaintext, 1.0)[0]),
        ('OURS', lambda: decode_constant(phantom.armor_extract_constant)),
        ('OPT_V1', lambda: decode_constant(phantom.extract_constant_optimize_v1)),
        ('OURS_OPT', lambda: decode_constant(phantom.extract_constant_optimize)),
    ]
    for _ in range(20):
        for index, (_, decode) in enumerate(paths):
            if index == 0:
                plaintext = sk.armor_decrypt(context, product, mult_key, example.galois_steps[0])
                phantom.cudaStreamSychronize()
            decode()
            phantom.cudaStreamSychronize()
    timings = [[] for _ in paths]
    for iteration in range(args.iterations):
        # Rotate execution order to reduce clock/temperature bias.
        order = [(iteration + offset) % len(paths) for offset in range(len(paths))]
        for index in order:
            # SOTA performs inverse NTT in place; refresh its input outside timing.
            if index == 0:
                plaintext = sk.armor_decrypt(context, product, mult_key, example.galois_steps[0])
            phantom.cudaStreamSychronize()
            start = time.perf_counter_ns()
            decoded = paths[index][1]()
            elapsed_us = (time.perf_counter_ns() - start) / 1000
            # Like armor_coeff.py, explicit synchronization is outside timing.
            phantom.cudaStreamSychronize()
            if not np.isclose(decoded, example.n, rtol=1e-9, atol=1e-6):
                raise AssertionError(f'{paths[index][0]}: {decoded} != {example.n}')
            timings[index].append(elapsed_us)
    print('Decode timing: SOTA = full coefficient decode; OURS/OURS_OPT = extract + constant decode')
    for (name, _), samples in zip(paths, timings):
        print(f'{name:8s}: median={statistics.median(samples)/1000:.6f} ms, '
              f'mean={statistics.mean(samples)/1000:.6f} ms, n={len(samples)}')
    print(f'OURS_OPT speedup vs OURS (median): '
          f'{statistics.median(timings[1])/statistics.median(timings[3]):.3f}x')
    print(f'OURS_OPT speedup vs OPT_V1 (median): '
          f'{statistics.median(timings[2])/statistics.median(timings[3]):.3f}x')



if __name__ == '__main__':
    main()
