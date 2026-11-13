"""One process runs every logical client sequentially and uploads encrypted deltas."""
import _shared  # Use Armor training, attacks and native library without copies.
import dataset
import train
from Ciphertext.encrypt_decrypt import encrypt
import options
import time
from distributed import (get_keys, connect, receive, seed_all, apply_update,
                         pack_ciphertexts, metered, log_event)


def main():

    local = options.args_parser()
    args, context, encoder, public_key, keys, spec = get_keys(local, 'client')
    seed_all(args.seed)
    dataset_train, _, _ = dataset.load_data(args)
    net_glob = dataset.init_model(args).to(args.device)
    log_event(args.log_path, 'client', 'config', extract_partial_constant=args.extract_partial_constant)

    with metered(connect(local.server1_host, local.server1_port, local.connect_timeout),
                 args, 'client', 'server1') as server1:
        server1.send(dict(type='client'))
        initial = receive(server1, 'initial', 0)
        net_glob.load_state_dict(initial['weights'])
        client_ids = list(range(args.num_clients))
        dict_users = initial['indices']
        previous = None
        for round_idx in range(args.global_ep):
            round_start = time.perf_counter()
            sent_before, received_before = server1.sent_bits, server1.received_bits
            training_timings = {}
            updates, train_time = train.update_clients(
                client_ids, dataset_train,
                dict_users, args, round_idx, initial['malicious_ids'], net_glob, previous,
                timings=training_timings)
            log_event(args.log_path, 'client', 'training', round=round_idx, latency_s=train_time,
                      timing_scope='lancelot_local_training_v1',
                      includes_model_copy=True, includes_loader_setup=True,
                      excludes_update_delta=True, excludes_flatten=True, excludes_cpu_transfer=True,
                      excludes_attack_postprocess=True)
            log_event(args.log_path, 'client', 'training_postprocess', round=round_idx,
                      latency_s=training_timings['postprocess_s'],
                      includes_update_delta=True, includes_attack_postprocess=True)
            enc_time = 0.0
            for client_id, update in zip(client_ids, updates):
                ciphertexts, elapsed = encrypt.encrypt_updates(
                    context, encoder, [update], public_key, spec['scale'], args)
                enc_time += elapsed
                log_event(args.log_path, 'client', 'encryption', round=round_idx, client_id=client_id, latency_s=elapsed)
                message = dict(type='local_update', round=round_idx, client_id=client_id,
                               ciphertexts=pack_ciphertexts(ciphertexts[0], log_path=args.log_path,
                                                           role='client', phase='upload', round_idx=round_idx))
                server1.send(message)
                del ciphertexts
            del updates
            message = receive(server1, 'global_update', round_idx)
            previous = message['update']
            apply_update(net_glob, previous)
            print(f'[Clients] round={round_idx + 1} count={args.num_clients} '
                  f'train={train_time:.4f}s encrypt={enc_time:.4f}s', flush=True)
            log_event(args.log_path, 'client', 'round_summary', round=round_idx,
                      latency_s=time.perf_counter() - round_start,
                      sent_frame_bits=server1.sent_bits - sent_before,
                      received_frame_bits=server1.received_bits - received_before)
            if message['done']:
                break


if __name__ == '__main__':
    main()
