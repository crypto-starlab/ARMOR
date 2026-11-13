"""Framing and fixed ARMOR parameters for the single-ciphertext TCP demo."""
import struct

HEADER = struct.Struct("!Q")
MAX_FRAME = 16 * 1024 * 1024
N = 16384
SCALE = 2.0 ** 74


def make_context(phantom):
    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, [62, 58, 58]))
    params.set_special_modulus_size(0)
    params.set_galois_elts(phantom.get_elts_from_steps([0], N))
    return phantom.context(params)


def recv_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(min(count - len(data), 65536))
        if not chunk:
            raise ConnectionError("Connection closed before complete frame arrived")
        data.extend(chunk)
    return bytes(data)


def send_frame(sock, payload):
    if not 0 < len(payload) <= MAX_FRAME:
        raise ValueError("Invalid frame size")
    sock.sendall(HEADER.pack(len(payload)) + payload)


def recv_frame(sock):
    count, = HEADER.unpack(recv_exact(sock, HEADER.size))
    if not 0 < count <= MAX_FRAME:
        raise ValueError("Invalid frame size")
    return recv_exact(sock, count)
