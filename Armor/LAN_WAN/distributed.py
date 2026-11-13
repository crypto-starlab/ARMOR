"""Shared ARMOR process transport and state helpers (trusted peers only)."""
import _shared  # Use Armor training, attacks and native library without copies.
import os
import json
import pickle
from pathlib import Path
import random
import time
from contextlib import contextmanager
from multiprocessing.connection import Client, Listener
from types import SimpleNamespace
import torch
import numpy as np
from Ciphertext.encrypt_decrypt import params
import pyPhantom as phantom
from collections import OrderedDict


@contextmanager
def crypto_timer():
    """Time completed native GPU work; callers keep RPC and serialization outside."""
    phantom.cudaStreamSychronize()
    timing = {}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        phantom.cudaStreamSychronize()
        timing['latency_s'] = time.perf_counter() - start


def log_decrypt_compute(log_path, role, phase, round_idx, timing, count):
    log_event(log_path, role, 'decrypt_compute', phase=phase, round=round_idx,
              latency_s=timing['latency_s'], count=count,
              includes_decode=role == 'server1', excludes_rpc=True,
              excludes_serialization=True)
    print(f'[{role}][round={round_idx + 1}][{phase}] '
          f'Dec compute={timing["latency_s"]:.6f}s '
          f'({"decrypt + decode" if role == "server1" else "partial decrypt"}; RPC excluded)',
          flush=True)

def setup_log(args, role):
    path = getattr(args, 'log_path', None)
    if not isinstance(path, str) or not path:
        mode = 'constant' if getattr(args, 'extract_partial_constant', True) else 'coefficient'
        protocol = ''
        path = f"{getattr(args, 'dataset', 'ARMOR')}_{getattr(args, 'attack', 'run')}{protocol}_{mode}_{role}.log"
    args.log_path = path.format(role=role)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    return args.log_path

def log_event(log_path, role, event, **fields):
    if log_path:
        record = dict(timestamp=time.time(), role=role, event=event, **fields)
        with open(log_path, 'a') as stream:
            stream.write(json.dumps(record, allow_nan=True) + '\n')

class MeteredConnection:
    """Measure serialized application frames, excluding auth and TCP/IP headers."""

    def __init__(self, connection, log_path, role, peer):
        self.connection, self.log_path = (connection, log_path)
        self.role, self.peer = (role, peer)
        self.sent_bits = self.received_bits = 0
        self._message_prefix = f'{role}-{os.getpid()}-{time.perf_counter_ns()}'
        self._message_sequence = 0

    def _record(self, direction, payload, message, elapsed, **timings):
        frame_bits = (len(payload) + (4 if len(payload) <= 2147483647 else 12)) * 8
        if direction == 'send':
            self.sent_bits += frame_bits
        else:
            self.received_bits += frame_bits
        fields = dict(peer=self.peer, direction=direction, message_type=message.get('type'), round=message.get('round'), phase=message.get('phase'), latency_s=elapsed, payload_bits=len(payload) * 8, frame_bits=frame_bits)
        fields.update(timings)
        fields['message_id'] = message.get('_armor_message_id')
        if 'ciphertexts' in message:
            inverse_bits = sum((len(x) * 8 for x in message.get('inverse_ciphertexts', [])))
            fields['serialized_ciphertext_bits'] = sum((len(x) * 8 for x in message['ciphertexts'])) + inverse_bits
            if inverse_bits:
                fields['serialized_inverse_ciphertext_bits'] = inverse_bits
        if 'partials' in message:
            parts = message['partials']
            fields['partial_representation'] = 'ciphertext' if parts and isinstance(parts[0], bytes) else 'constant'
            fields['partial_data_bits'] = sum((len(x) * (8 if isinstance(x, bytes) else 64) for x in parts))
        log_event(self.log_path, self.role, 'communication', **fields)
        if direction == 'send':
            print(f'[{self.role}->{self.peer}] round={message.get("round")} '
                  f'phase={message.get("phase")} type={message.get("type")} '
                  f'pickle={timings["serialize_s"]:.6f}s '
                  f'blocking_send={timings["blocking_send_s"]:.6f}s', flush=True)

    def send(self, message):
        self._message_sequence += 1
        message = dict(message, _armor_message_id=f'{self._message_prefix}-{self._message_sequence}')
        start = time.perf_counter()
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        serialized = time.perf_counter()
        send_start_ns = time.perf_counter_ns()
        self.connection.send_bytes(payload)
        sent = time.perf_counter()
        self._record('send', payload, message, sent - start,
                     serialize_s=serialized - start, blocking_send_s=sent - serialized,
                     send_start_ns=send_start_ns)

    def recv(self):
        start = time.perf_counter()
        payload = self.connection.recv_bytes()
        receive_complete_ns = time.perf_counter_ns()
        received = time.perf_counter()
        message = pickle.loads(payload)
        decoded = time.perf_counter()
        if message.get('type') == 'keys':
            self.log_path = setup_log(SimpleNamespace(**message['args']), self.role)
        self._record('receive', payload, message, decoded - start,
                     blocking_receive_s=received - start, deserialize_s=decoded - received,
                     receive_complete_ns=receive_complete_ns)
        message.pop('_armor_message_id', None)
        return message

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def metered(connection, args, role, peer):
    return MeteredConnection(connection, args.log_path, role, peer)

def authkey():
    return os.environ.get('ARMOR_AUTHKEY', 'armor-local-development').encode()

def listen(host, port):
    return Listener((host, port), authkey=authkey(), backlog=128)

def connect(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return Client((host, port), authkey=authkey())
        except (ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)

def receive(conn, kind, round_idx=None):
    message = conn.recv()
    return message

def configure_device(args):
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    return args

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parameter_spec(args):
    version1 = args.dataset in ('CIFAR10', 'FMNIST')
    suffix = 'version1' if version1 else 'version2'
    return dict(ringdim=args.ringdim, modulus_chain=list(params.modulus_chain_ver1 if version1 else params.modulus_chain_ver2), galois_steps=list(params.galois_steps), size_P=params.size_P, scale=getattr(params, 'scale_' + suffix), is_version2=not version1)

def make_context(spec):
    p = phantom.params(phantom.scheme_type.ckks)
    p.set_poly_modulus_degree(spec['ringdim'])
    p.set_coeff_modulus(phantom.create_coeff_modulus(spec['ringdim'], spec['modulus_chain']))
    p.set_special_modulus_size(spec['size_P'])
    p.set_galois_elts(phantom.get_elts_from_steps(spec['galois_steps'], spec['ringdim']))
    return phantom.context(p)

def get_keys(local_args, role):
    with metered(connect(local_args.keycenter_host, local_args.keycenter_port, local_args.connect_timeout), local_args, role, 'keycenter') as conn:
        conn.send(dict(type='hello', role=role))
        packet = receive(conn, 'keys')
    args = SimpleNamespace(**packet['args'])
    args.gpu = local_args.gpu
    configure_device(args)
    setup_log(args, role)
    log_event(args.log_path, role, 'run_config', dataset=args.dataset,
              global_ep=args.global_ep, num_clients=args.num_clients)
    print(f'[{role}] KeyCenter config: dataset={args.dataset} '
          f'global_ep={args.global_ep} clients={args.num_clients}', flush=True)
    context = make_context(packet['params'])
    pk = phantom.public_key.from_bytes(packet['public_key'])
    keys = {name: phantom.secret_key.from_bytes(packet[name]) for name in ('sk1', 'sk1_remain', 'sk2', 'sk2_remain') if name in packet}
    return (args, context, phantom.ckks_encoder(context), pk, keys, packet['params'])

def cpu_state(state):
    return OrderedDict(((k, v.detach().cpu().clone()) for k, v in state.items()))

def apply_update(model, update):
    state = model.state_dict()
    for key in state:
        state[key] += update[key].to(device=state[key].device, dtype=state[key].dtype)
    model.load_state_dict(state)

def mad_filter(values, candidates=None, threshold=3.5):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    candidates = list(range(len(values))) if candidates is None else list(candidates)
    if mad == 0:
        return (candidates, median)
    scores = 0.6745 * np.abs(values - median) / mad
    return ([i for i in candidates if scores[i] <= threshold], median)

def pack_ciphertexts(ciphertexts, *, log_path=None, role=None, phase=None, round_idx=None):
    start = time.perf_counter()
    blobs = [ct.to_bytes() for ct in ciphertexts]
    elapsed = time.perf_counter() - start
    if log_path:
        log_event(log_path, role, 'ciphertext_serialization', phase=phase,
                  round=round_idx, latency_s=elapsed, count=len(blobs))
        print(f'[{role}] round={round_idx} phase={phase} '
              f'ciphertext_serialize={elapsed:.6f}s', flush=True)
    return blobs

def unpack_ciphertexts(blobs):
    return [phantom.ciphertext.from_bytes(blob) for blob in blobs]

def remote_partial(conn, phase, round_idx, ciphertexts):
    start = time.perf_counter()
    blobs = pack_ciphertexts(ciphertexts, log_path=getattr(conn, 'log_path', None),
                             role='server1', phase=phase, round_idx=round_idx)
    conn.send(dict(type='partial', phase=phase, round=round_idx, ciphertexts=blobs))
    response = receive(conn, 'partial_result', round_idx)
    log_event(getattr(conn, 'log_path', None), 'server1', 'partial_rpc', phase=phase, round=round_idx, round_trip_latency_s=time.perf_counter() - start)
    return response['partials']
