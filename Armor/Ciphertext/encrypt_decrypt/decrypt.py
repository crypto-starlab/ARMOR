import pyPhantom as phantom
import numpy as np
import torch
import time
def decrypt_layer(context, encoder, ciphertext, sk1, sk2, decrypt_similarity, len):
    
    ciphertext = phantom.armor_upscale_and_rescale(context, phantom.armor_to_intt(context, ciphertext))
    partial = []
    ciphertext = phantom.armor_to_ntt(context, ciphertext)
    ciphertextBits = phantom.getCiphertextBits(ciphertext)
    partial1 = sk1.armor_individual_decrypt(context, ciphertext, decrypt_similarity)
    partial2 = sk2.armor_individual_decrypt_last(context, ciphertext, decrypt_similarity)
   
    partial.append(partial1)
    partial.append(partial2)
    partialBits = phantom.getCiphertextBits(partial2)
    
    full_decrypt = sk1.armor_multiparty_decrypt(context, partial)

    res = encoder.armor_coefficient_decode_and_mult(context, full_decrypt, len)

    return res, ciphertextBits, partialBits
    

def decrypt_global_update(context, encoder, encrypted_global_update, sk1, sk2, aggregate_len, log_path):
    start_agg_dec = time.time()
    aggregate_update = []
    ciphertextBits = 0
    partialBits = 0
    for i in range(len(encrypted_global_update)):
        partial_update, bits0, bits1  = decrypt_layer(context, encoder, encrypted_global_update[i], sk1, sk2, False, aggregate_len)
        aggregate_update.append(partial_update)
        ciphertextBits += bits0
        partialBits += bits1
    t_agg_dec = time.time() - start_agg_dec

    '''
    ciphertextBits : Sa -> Ss(aggregation ciphertexts) only d1 poly
    partialBits : Ss -> Sa(partial decrypt cipehrtexts)
    '''

    with open(log_path, "a") as f:
        f.write(f"Sa-> Ss aggregation ciphertexts(only d1) : {ciphertextBits/(8 * 1024*1024*2)}MB\n")
        f.write(f"Ss-> Sa partialDecrypt ciphertexts : {partialBits/(8 * 1024*1024)}MB\n")
    return aggregate_update, t_agg_dec
    
def merge_chunks(chunk_list, original_len):
    merged = np.concatenate(chunk_list, axis=0)
    return torch.tensor(merged[:original_len])

def get_param_shapes(model):
    shapes = []
    names = []
    for name, param in model.state_dict().items():
        shapes.append(param.shape)
        names.append(name)
    return names, shapes


def vector_to_state_dict(vec, param_names, param_shapes, device):
    new_state = {}
    idx = 0
    for name, shape in zip(param_names, param_shapes):
        numel = int(torch.prod(torch.tensor(shape)).item())
        new_state[name] = vec[idx:idx+numel].reshape(shape).to(device)
        idx += numel
    return new_state


def update_global_model(net_glob, aggregate_update, orig_len, args):
    merged_vec = merge_chunks(aggregate_update, orig_len)
    param_names, param_shapes = get_param_shapes(net_glob)
    global_update = vector_to_state_dict(merged_vec, param_names, param_shapes, args.device)
        
    global_weights = net_glob.state_dict()
    for key in global_update.keys():
        global_update[key] = global_update[key].to(
            device=global_weights[key].device, dtype=global_weights[key].dtype)
        global_weights[key] += global_update[key]

    net_glob.load_state_dict(global_weights)
    return global_update
