"""Server2 holds sk2 and provides partial decryption to Server1."""
import _shared  # Use Armor training, attacks and native library without copies.
import options
import time
import pyPhantom as phantom
from distributed import get_keys, listen, receive, unpack_ciphertexts, pack_ciphertexts, metered, log_event, crypto_timer, log_decrypt_compute


def partial_decrypt(context, keys, spec, phase, ciphertexts, extract_partial_constant=True, timing_out=None,
                    log_path=None, round_idx=None):
    with crypto_timer() as timing:
        partials = []
        for ct in ciphertexts:
            if phase == 'norm':
                partial = keys['sk2'].armor_individual_decrypt_last(context, ct, keys['sk2_remain'], spec['galois_steps'][0],
                                                                        extract_partial_constant, spec['is_version2'])
            elif phase == 'similarity':
                partial = keys['sk2'].armor_individual_decrypt_last(context, ct, extract_partial_constant,
                                                                        False, spec['is_version2'])
            else:
                partial = keys['sk2'].armor_individual_decrypt_last(context, ct, False)
            if phase != 'aggregation' and extract_partial_constant:
                partial = phantom.armor_extract_partial_constant(context, partial, phase == 'norm', spec['is_version2'])
            partials.append(partial)
    if timing_out is not None:
        timing_out.update(timing)
    if phase == 'aggregation' or not extract_partial_constant:
        partials = pack_ciphertexts(partials, log_path=log_path, role='server2',
                                   phase=phase, round_idx=round_idx)
    phantom.cudaStreamSychronize()
    return partials


def main():
    local = options.args_parser()
    args, context, encoder, public_key, keys, spec = get_keys(local, 'server2')
    with listen(local.server2_host, local.server2_port) as listener:
        print(f'[Server2] listening on {listener.address}', flush=True)
        with metered(listener.accept(), args, 'server2', 'server1') as conn:
            receive(conn, 'server1')
            while True:
                message = conn.recv()
                if message.get('type') == 'stop':
                    break
                phase, round_idx = message.get('phase'), message.get('round')
                start = time.perf_counter()
                ciphertexts = unpack_ciphertexts(message['ciphertexts'])
                timing = {}
                partials = partial_decrypt(context, keys, spec, phase, ciphertexts,
                                          args.extract_partial_constant, timing_out=timing,
                                          log_path=args.log_path, round_idx=round_idx)
                log_decrypt_compute(args.log_path, 'server2', phase, round_idx, timing, len(partials))
                log_event(args.log_path, 'server2', 'partial_decrypt', phase=phase, round=round_idx,
                          extract_partial_constant=args.extract_partial_constant,
                          latency_s=time.perf_counter() - start, count=len(partials))
                conn.send(dict(type='partial_result', phase=phase, round=round_idx, partials=partials))
                print(f'[Server2] round={round_idx + 1} phase={phase} count={len(partials)}', flush=True)


if __name__ == '__main__':
    main()
