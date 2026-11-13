"""Server1 initializes weights, filters encrypted updates, and broadcasts their mean."""
import _shared  # Use Armor training, attacks and native library without copies.
import math
import random
import time
import options
from distributed import (get_keys, seed_all, listen, connect, receive, cpu_state,
                         unpack_ciphertexts, remote_partial, mad_filter, metered, log_event,
                         crypto_timer, log_decrypt_compute)
import dataset
import sampling
import test
from Ciphertext.encrypt_decrypt import encrypt, decrypt
from Ciphertext.aggregation import aggregation
import pyPhantom as phantom
import numpy as np
from Ciphertext import utils
from Plaintext import utils as plain_utils
import torch
def decrypt_statistics(conn, context, spec, keys, phase, round_idx, ciphertexts, denominators=None, encoder=None, extract_partial_constant=True):
    
    start = time.perf_counter()
    remote = remote_partial(conn, phase, round_idx, ciphertexts)
    if not extract_partial_constant:
        remote = unpack_ciphertexts(remote)
        if encoder is None:
            encoder = phantom.ckks_encoder(context)
    with crypto_timer() as timing:
        values = []
        for i, ct in enumerate(ciphertexts):
            if phase == 'norm':
                partial = keys['sk1'].armor_individual_decrypt(context, ct, keys['sk1_remain'], spec['galois_steps'][0],
                                                                   extract_partial_constant, spec['is_version2'])
            else:
                partial = keys['sk1'].armor_individual_decrypt(context, ct, extract_partial_constant,
                                                                   False, spec['is_version2'])
            divisor = 1.0 if denominators is None else denominators[i]
            if extract_partial_constant:
                partial = phantom.armor_extract_partial_constant(context, partial, phase == 'norm', spec['is_version2'])
                value = phantom.armor_decode_constant(context, partial, remote[i], spec['scale'], divisor)
            else:
                plaintext = keys['sk1'].armor_multiparty_decrypt(context, [partial, remote[i]])
                decoded = encoder.armor_coefficient_decode_and_mult(context, plaintext, 1.0)
                value = float(decoded[0]) / divisor
            values.append(value)
    log_decrypt_compute(getattr(conn, 'log_path', None), 'server1', phase,
                        round_idx, timing, len(values))
    if phase == 'norm':
        values = [math.sqrt(max(value, 0) + 1e-5) for value in values]
    log_event(getattr(conn, 'log_path', None), 'server1', 'statistics_decrypt',
              phase=phase, round=round_idx, extract_partial_constant=extract_partial_constant,
              latency_s=time.perf_counter() - start, count=len(values))
    return values


def decrypt_aggregate(conn, context, encoder, keys, round_idx, ciphertexts, count):
    start = time.perf_counter()
    prepared = [phantom.armor_to_ntt(context, phantom.armor_upscale_and_rescale(
        context, phantom.armor_to_intt(context, ct))) for ct in ciphertexts]
    remote = unpack_ciphertexts(remote_partial(conn, 'aggregation', round_idx, prepared))
    with crypto_timer() as timing:
        result = []
        for ct, partial2 in zip(prepared, remote):
            partial1 = keys['sk1'].armor_individual_decrypt(context, ct, False)
            plaintext = keys['sk1'].armor_multiparty_decrypt(context, [partial1, partial2])
            decoded = encoder.armor_coefficient_decode_and_mult(context, plaintext, count)
            result.append(decoded)
    log_decrypt_compute(getattr(conn, 'log_path', None), 'server1', 'aggregation',
                        round_idx, timing, len(result))
    log_event(getattr(conn, 'log_path', None), 'server1', 'aggregate_decrypt',
              round=round_idx, latency_s=time.perf_counter() - start, count=len(result))
    return result


def filter_updates(conn, context, encoder, spec, keys, args, round_idx, ciphertexts, previous):
    reference = utils.global_update_linear(previous, args)
    norm_ciphertexts, norm_he_time = utils.Norm_ciphertexts(context, args, ciphertexts,
                                               spec['galois_steps'][0], len(spec['modulus_chain']))
    log_event(args.log_path, 'server1', 'norm_he', round=round_idx, latency_s=norm_he_time)
    norms = decrypt_statistics(conn, context, spec, keys, 'norm', round_idx, norm_ciphertexts,
                               encoder=encoder, extract_partial_constant=args.extract_partial_constant)
    first, _ = mad_filter(norms)
    global_norm = float(plain_utils.get_norm_chunk(reference, device=args.device))
    if global_norm <= 1e-12:
        return first, norms, [0.0] * len(norms)
    encoded = encrypt.encode_global_updates_conjugated(encoder, context, reference, spec['scale'])
    sim_ciphertexts, sim_he_time = utils.Similarity_ciphertexts(context, args, ciphertexts,
                                                    encoded, len(spec['modulus_chain']))
    log_event(args.log_path, 'server1', 'similarity_he', round=round_idx, latency_s=sim_he_time)
    # Preserve the original ARMOR statistic: dot / local_norm; global_norm is common.
    similarities = decrypt_statistics(conn, context, spec, keys, 'similarity', round_idx,
                                      sim_ciphertexts, [n + 1e-8 for n in norms],
                                      encoder=encoder, extract_partial_constant=args.extract_partial_constant)
    survivors, _ = mad_filter(similarities, first)
    return survivors, norms, list(np.asarray(similarities) / global_norm)


def main():
    local = options.args_parser()
    args, context, encoder, public_key, keys, spec = get_keys(local, 'server1')
    seed_all(args.seed)
    dataset_train, dataset_test, _ = dataset.load_data(args)
    dict_users = sampling.noniid(dataset_train, args)
    net_glob = dataset.init_model(args).to(args.device)
    original_len = encrypt.flatten_weight(net_glob).numel()
    chunk_count = (original_len + args.ringdim - 1) // args.ringdim
    malicious = set(random.sample(range(args.num_clients), max(int(args.attack_ratio * args.num_clients), 1))) if args.attack_ratio > 0 else set()
    log_event(args.log_path, 'server1', 'config', extract_partial_constant=args.extract_partial_constant)
    previous = None
    ids = list(range(args.num_clients))
    with metered(connect(local.server2_host, local.server2_port, local.connect_timeout),
                 args, 'server1', 'server2') as server2:
        server2.send(dict(type='server1'))
        with listen(local.server1_host, local.server1_port) as listener:
            print(f'[Server1] waiting for the sequential client process on {listener.address}', flush=True)
            with metered(listener.accept(), args, 'server1', 'client') as client:
                receive(client, 'client')
                client.send(dict(type='initial', round=0, weights=cpu_state(net_glob.state_dict()),
                                 indices={i: sorted(dict_users[i]) for i in ids},
                                 malicious_ids=sorted(malicious)))
                for round_idx in range(args.global_ep):
                    round_start = time.perf_counter()
                    sent_before = server2.sent_bits + client.sent_bits
                    received_before = server2.received_bits + client.received_bits
                    ciphertexts = []
                    for client_id in ids:
                        message = receive(client, 'local_update', round_idx)
                        ciphertexts.append(unpack_ciphertexts(message['ciphertexts']))
                    survivors, norms, similarities = list(range(len(ids))), [], []
                    if previous is not None and args.method == 'armor':
                        survivors, norms, similarities = filter_updates(
                            server2, context, encoder, spec, keys, args, round_idx, ciphertexts, previous)
                    if survivors:
                        summed, agg_he_time = aggregation.aggregation_sum_last_survivor(context, ciphertexts, survivors)
                        log_event(args.log_path, 'server1', 'aggregation_he', round=round_idx, latency_s=agg_he_time)
                        decoded = decrypt_aggregate(server2, context, encoder, keys, round_idx, summed, len(survivors))
                        previous = decrypt.update_global_model(net_glob, decoded, original_len, args)
                    else:
                        previous = {k: torch.zeros_like(v) for k, v in net_glob.state_dict().items()}
                    acc, loss = test.test_img(net_glob, dataset_test, round_idx, args)
                    log_event(args.log_path, 'server1', 'evaluation', round=round_idx,
                              client_ids=ids, survivor_ids=[ids[i] for i in survivors],
                              norms=norms, cosines=similarities, accuracy=float(acc), loss=float(loss))
                    print(f'[Server1] round={round_idx + 1} survivors={len(survivors)} acc={acc:.4f} loss={loss:.4f}', flush=True)
                    client.send(dict(type='global_update', round=round_idx, update=cpu_state(previous),
                                     done=round_idx + 1 == args.global_ep))
                    log_event(args.log_path, 'server1', 'round_summary', round=round_idx,
                              latency_s=time.perf_counter() - round_start,
                              sent_frame_bits=server2.sent_bits + client.sent_bits - sent_before,
                              received_frame_bits=server2.received_bits + client.received_bits - received_before)
                server2.send(dict(type='stop'))


if __name__ == '__main__':
    main()
