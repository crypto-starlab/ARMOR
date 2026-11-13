"""Encrypt one ARMOR chunk, send it, and verify the restored ciphertext echo."""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import pyPhantom as phantom
from armor_tcp_common import N, SCALE, HEADER, make_context, send_frame, recv_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--output", type=Path, help="Optionally save serialized ciphertext")
    args = parser.parse_args()
    context = make_context(phantom)
    encoder = phantom.ckks_encoder(context)
    sk = phantom.secret_key(context)
    pk = sk.gen_publickey(context)
    values = np.random.default_rng(2023).uniform(-0.1, 0.1, N)
    pt = encoder.armor_coefficient_encode(context, values, SCALE, 0)
    ct = pk.armor_encrypt_asymmetric(context, pt)
    phantom.cudaStreamSychronize()
    start = time.perf_counter()
    payload = ct.to_bytes()
    serialize_ms = (time.perf_counter() - start) * 1000
    if args.output:
        args.output.write_bytes(payload)
        assert args.output.read_bytes() == payload

    with socket.create_connection((args.host, args.port), timeout=60) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        start = time.perf_counter()
        send_frame(sock, b"ping")
        assert recv_frame(sock) == b"pong"
        rtt_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        send_frame(sock, payload)
        # Server acknowledges complete receipt BEFORE GPU restoration.
        receipt = json.loads(recv_frame(sock))
        upload_ack_s = time.perf_counter() - start
        restored_payload = recv_frame(sock)
        assert restored_payload == payload, "Server restore/reserialize changed bytes"

    restored = phantom.ciphertext.from_bytes(restored_payload)
    decoded = encoder.armor_coefficient_decode_and_mult(
        context, sk.decrypt(context, restored), 1.0
    )
    np.testing.assert_allclose(decoded, values, atol=1e-6, rtol=0)
    print(json.dumps({
        "verified": True,
        "payload_bytes": len(payload),
        "upload_frame_bytes": len(payload) + HEADER.size,
        "serialize_ms": serialize_ms,
        "application_rtt_ms": rtt_ms,
        "upload_ack_ms": upload_ack_s * 1000,
        "upload_ack_goodput_mbps": len(payload) * 8 / upload_ack_s / 1e6,
        "max_decrypt_error": float(np.max(np.abs(np.asarray(decoded) - values))),
        "server": receipt,
    }, indent=2))


if __name__ == "__main__":
    main()
