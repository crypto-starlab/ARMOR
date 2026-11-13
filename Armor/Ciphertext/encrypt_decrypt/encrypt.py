import pyPhantom as phantom
import torch
from collections import OrderedDict
import time
import numpy as np

def flatten_global_update(state_dict):
    vec = []
    for k in state_dict:
        vec.append(state_dict[k].reshape(-1))
    return torch.cat(vec)

def flatten_weight(net):
    if hasattr(net, 'state_dict'):
        state = net.state_dict()
    elif isinstance(net, OrderedDict):
        state = net
    vec = []
    for k in state:
        vec.append(state[k].reshape(-1).detach().cpu())
    return torch.cat(vec)

def dividing_by_chunk(grad_list, args):
    chunk_size = args.ringdim
    total_len = grad_list.numel()
    pad_len = (chunk_size - total_len % chunk_size) % chunk_size
    if pad_len > 0:
        grad_list = torch.nn.functional.pad(grad_list, (0, pad_len))
    chunk_list = list(grad_list.view(-1, chunk_size))
    return chunk_list

def encrypt_local_updates(encoder, context, chunk_list, public_key, scale):
    ciphertexts = []
    for chunk in chunk_list:
        chunk_np = chunk.detach().cpu().numpy()
        plaintext_chunk = encoder.armor_coefficient_encode(context, chunk_np, scale, 0)
        ciphertexts.append(public_key.armor_encrypt_asymmetric(context, plaintext_chunk))
    return ciphertexts

def conjugate_rotate_form_in_list(global_chunk_list):
    """f(X^-1) mod (X^N+1): [a0, -a(N-1), ..., -a1], not 1/a_i."""
    result = []
    for chunk in global_chunk_list:
        chunk_np = chunk.detach().cpu().numpy() if torch.is_tensor(chunk) else np.asarray(chunk)
        if chunk_np.ndim != 1 or chunk_np.size == 0 or (not np.isfinite(chunk_np).all()):
            raise ValueError('Expected a nonempty finite coefficient vector')
        copy_chunk = chunk_np.copy()
        copy_chunk[1:] = -chunk_np[:0:-1]
        result.append(copy_chunk)
    return result

def encode_global_updates_conjugated(encoder, context, global_chunk_list, scale):
    global_chunk_list_conjugated = conjugate_rotate_form_in_list(global_chunk_list)
    encoded_global_updates = []
    for chunk in global_chunk_list_conjugated:
        plaintext_chunk = encoder.armor_coefficient_encode_ntt(context, chunk, scale, 0)
        encoded_global_updates.append(plaintext_chunk)
    return encoded_global_updates

def flatten_then_encrypt_updates(net, context, encoder, public_key, scale, args):
    flatten_vector = dividing_by_chunk(flatten_weight(net), args)
    encrypted_vector = encrypt_local_updates(encoder, context, flatten_vector, public_key, scale)
    return encrypted_vector

def encrypt_updates(context, encoder, update_list, public_key, scale, args):
    start_enc = time.time()
    encrypted_local_updates = []
    for update in update_list:
        encrypted_local_updates.append(flatten_then_encrypt_updates(update, context, encoder, public_key, scale, args))
    phantom.cudaStreamSychronize()
    t_enc = time.time() - start_enc
    return (encrypted_local_updates, t_enc)
