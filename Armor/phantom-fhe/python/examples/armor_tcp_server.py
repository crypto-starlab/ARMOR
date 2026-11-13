"""Receive one ciphertext, acknowledge receipt, restore, and echo serialized data."""
import argparse
import json
import socket
import time
from pathlib import Path

import pyPhantom as phantom
from armor_tcp_common import make_context, send_frame, recv_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--output", type=Path, help="Optionally save received ciphertext")
    args = parser.parse_args()
    context = make_context(phantom)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind, args.port))
        listener.listen(1)
        print(f"Listening on {args.bind}:{args.port}", flush=True)
        sock, address = listener.accept()
        with sock:
            sock.settimeout(60)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if recv_frame(sock) != b"ping":
                raise ValueError("Expected ping")
            send_frame(sock, b"pong")
            payload = recv_frame(sock)
            send_frame(sock, json.dumps({"received_bytes": len(payload)}).encode())
            if args.output:
                args.output.write_bytes(payload)
                payload = args.output.read_bytes()
            start = time.perf_counter()
            ct = phantom.ciphertext.from_bytes(payload)
            phantom.cudaStreamSychronize()
            restore_ms = (time.perf_counter() - start) * 1000
            echoed = ct.to_bytes()
            if echoed != payload:
                raise RuntimeError("Restore/reserialize mismatch")
            send_frame(sock, echoed)
            print(json.dumps({"client": address, "received_bytes": len(payload),
                              "restore_ms": restore_ms, "byte_roundtrip_verified": True}))


if __name__ == "__main__":
    main()
