"""Generate ARMOR parameters and distribute only each recipient's key shares."""
import _shared  # Use Armor training, attacks and native library without copies.
import options
import time
from distributed import configure_device, parameter_spec, listen, receive, setup_log, metered, log_event
from Ciphertext.encrypt_decrypt import params

def key_packet(role, config, spec, public_key, shares):
    packet = dict(type='keys', args=config, params=spec, public_key=public_key)
    if role == 'server1':
        packet.update(sk1=shares['sk1'], sk1_remain=shares['sk1_remain'])
    elif role == 'server2':
        packet.update(sk2=shares['sk2'], sk2_remain=shares['sk2_remain'])
    return packet


def main():
    args = configure_device(options.args_parser())
    if args.norm_protocol != 'armor':
        raise ValueError('Base FL roles support ARMOR only; use the method folders for norm comparisons')
    args.cipher = True
    log_template = args.log_path
    setup_log(args, 'keycenter')
    log_event(args.log_path, 'keycenter', 'run_config', dataset=args.dataset,
              global_ep=args.global_ep, num_clients=args.num_clients)
    print(f'[KeyCenter] config: dataset={args.dataset} global_ep={args.global_ep} '
          f'clients={args.num_clients}', flush=True)
    start = time.perf_counter()
    context, *_ = params.set_context(args)
    sk1, sk2, sk1_remain, sk2_remain, public_key, _ = params.set_key_setting(context, args.norm_protocol)
    shares = {name: key.to_bytes() for name, key in
              [('sk1', sk1), ('sk2', sk2), ('sk1_remain', sk1_remain), ('sk2_remain', sk2_remain)]}
    log_event(args.log_path, 'keycenter', 'key_generation', latency_s=time.perf_counter() - start)
    config = {k: v for k, v in vars(args).items() if k != 'device'}
    config['log_path'] = log_template
    spec, public_bytes = parameter_spec(args), public_key.to_bytes()
    expected = {'server1', 'server2', 'client'}
    with listen(args.keycenter_host, args.keycenter_port) as listener:
        print(f'[KeyCenter] listening on {listener.address}', flush=True)
        while expected:
            with metered(listener.accept(), args, 'keycenter', 'participant') as conn:
                hello = receive(conn, 'hello')
                role = hello.get('role')
                identity = role
                conn.peer = role
                if identity not in expected:
                    conn.send(dict(type='error', message='Unknown or duplicate participant'))
                    continue
                conn.send(key_packet(role, config, spec, public_bytes, shares))
                expected.remove(identity)
                print(f'[KeyCenter] keys sent to {identity}', flush=True)


if __name__ == '__main__':
    main()
