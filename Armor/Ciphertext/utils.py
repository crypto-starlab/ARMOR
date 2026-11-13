import pyPhantom as phantom
import math
import numpy as np
import torch
from Ciphertext.encrypt_decrypt import encrypt
from Plaintext.intersection import filter_sets
from armor_filter_config import STD_THRESHOLD
import time
eps_norm = 1e-05
eps_sim = 1e-08

def global_update_linear(global_update, args):
    global_update_list = encrypt.flatten_global_update(global_update)
    global_update_chunks = encrypt.dividing_by_chunk(global_update_list, args)
    return global_update_chunks

def total_norm_from_chunks(chunks, p=2):
    total_sq = 0.0
    for chunk in chunks:
        total_sq += torch.sum(chunk ** 2)
    return torch.sqrt(total_sq)

def get_norm_squared_from_chunks_plaintext(chunks):
    norm_squared = 0.0
    for chunk in chunks:
        chunk_np = chunk.detach().cpu().numpy().tolist()
        for val in chunk_np:
            norm_squared += val * val
    return norm_squared

def normalize_chunks_by_total_norm(chunks):
    total_norm = total_norm_from_chunks(chunks)
    if total_norm == 0.0:
        print('invalid norm!')
    normalized_chunks = [chunk / total_norm for chunk in chunks]
    return (normalized_chunks, total_norm)

def conjugate_rotate_clients_ciphertext(context, step, client_ciphertexts):
    rotate_ciphers = []
    for i in range(0, len(client_ciphertexts)):
        rotate_ciphers.append(phantom.rotate_without_key_switching(context, client_ciphertexts[i], step))
    return rotate_ciphers

def multiply_and_sum_ciphertexts_similarity(context, ciphertext_list, global_update_plain_list, modulus_len):
    res = None
    for i in range(0, len(ciphertext_list)):
        if res == None:
            res = phantom.armor_multiply_plain(context, ciphertext_list[i], global_update_plain_list[i])
        else:
            res = phantom.armor_add(context, res, phantom.armor_multiply_plain(context, ciphertext_list[i], global_update_plain_list[i]))
    res = phantom.armor_to_intt(context, res)
    res = phantom.armor_rescale_to(context, res, modulus_len - 2)
    res = phantom.armor_to_ntt(context, res)
    return res

def multiply_with_rot_and_sum_ciphertexts(context, ciphertext_list1, step, modulus_len):
    res = None
    for i in range(0, len(ciphertext_list1)):
        rotated_cipher = phantom.armor_rotate_without_key_switching(context, ciphertext_list1[i], step)
        if res == None:
            res = phantom.armor_multiply(context, ciphertext_list1[i], rotated_cipher)
        else:
            res = phantom.armor_add(context, res, phantom.armor_multiply(context, ciphertext_list1[i], rotated_cipher))
    res = phantom.armor_to_intt(context, res)
    res = phantom.armor_rescale_to(context, res, modulus_len - 2)
    res = phantom.armor_to_ntt(context, res)
    return res

def norm_check(context, scale, ciphertext, sk1, sk1_remain, sk2, sk2_remain, step, is_norm, is_version2, norm_protocol='armor'):
    partial1 = sk1.armor_individual_decrypt(context, ciphertext, sk1_remain, step)
    partial2 = sk2.armor_individual_decrypt_last(context, ciphertext, sk2_remain, step)
    s1_generate_partial_constant = phantom.armor_extract_partial_constant(context, partial1, is_norm, is_version2)
    s2_generate_partial_constant = phantom.armor_extract_partial_constant(context, partial2, is_norm, is_version2)
    partialNormBits = phantom.getPartialBits(s2_generate_partial_constant)
    return (phantom.armor_decode_constant(context, s1_generate_partial_constant, s2_generate_partial_constant, scale, 1.0), partialNormBits)

def similarity_check(context, scale, scalar, ciphertext, sk1, sk2, is_similarity, is_norm, is_version2):
    partial = []
    partial1 = sk1.armor_individual_decrypt(context, ciphertext, is_similarity)
    partial2 = sk2.armor_individual_decrypt_last(context, ciphertext, is_similarity)
    s1_generate_partial_constant = phantom.armor_extract_partial_constant(context, partial1, is_norm, is_version2)
    s2_generate_partial_constant = phantom.armor_extract_partial_constant(context, partial2, is_norm, is_version2)
    partialSimilairtyBits = phantom.getPartialBits(s2_generate_partial_constant)
    return (phantom.armor_decode_constant(context, s1_generate_partial_constant, s2_generate_partial_constant, scale, scalar), partialSimilairtyBits)

def norm_check_using_modified_z_score(norm_lists, z_thresh=STD_THRESHOLD):
    values = np.asarray(norm_lists, dtype=np.float64)
    mad_ids, std_ids = filter_sets(values, std_threshold=z_thresh)
    return sorted(mad_ids & std_ids), np.median(values)


def similarity_check_using_modified_z_score(client_wise_similarity, first_remain, z_thresh=STD_THRESHOLD):
    values = np.asarray(client_wise_similarity, dtype=np.float64)
    mad_ids, std_ids = filter_sets(values, similarity=True, std_threshold=z_thresh)
    survivors = sorted(set(first_remain) & mad_ids & std_ids)
    return survivors, torch.tensor(np.sort(values)[(len(values) - 1) // 2])

def Norm_ciphertexts(context, args, ciphertexts, step, modulus_len, inverse_ciphertexts=None):
    start_norm_he = time.time()
    client_wise_norm_ciphertexts = []
    for id in range(0, args.num_clients):
        client_ciphertexts = ciphertexts[id]
        client_norm_ciphertext = multiply_with_rot_and_sum_ciphertexts(context, client_ciphertexts, step, modulus_len)
        client_wise_norm_ciphertexts.append(client_norm_ciphertext)
    phantom.cudaStreamSychronize()
    t_norm_he = time.time() - start_norm_he
    return (client_wise_norm_ciphertexts, t_norm_he)

def decrypt_norm_ciphertexts(context, args, scale, client_wise_norm_ciphertexts, sk1, sk1_remain, sk2, sk2_remain, step, is_version2, log_path):
    start_norm_dec = time.time()
    client_wise_norm = []
    partialBits = 0
    for id in range(0, args.num_clients):
        client_norm_ciphertext = client_wise_norm_ciphertexts[id]
        norm_squared, bits = norm_check(context, scale, client_norm_ciphertext, sk1, sk1_remain, sk2, sk2_remain, step, True, is_version2, norm_protocol=getattr(args, 'norm_protocol', 'armor'))
        partialBits += bits
        client_norm = math.pow(norm_squared + eps_norm, 0.5)
        client_wise_norm.append(client_norm)
    t_norm_dec = time.time() - start_norm_dec
    with open(log_path, 'a') as f:
        f.write(f'Ss -> Sa partial norm : {partialBits / (8 * 1024.0 * 1024.0)}MB\n')
    return (client_wise_norm, t_norm_dec)

def client_wise_norm_phase(context, args, scale, ciphertexts, modulus_len, step, sk1, sk1_remain, sk2, sk2_remain, is_version2, log_path, inverse_ciphertexts=None):
    client_wise_norm_ciphertexts, t_norm_he = Norm_ciphertexts(context, args, ciphertexts, step, modulus_len, inverse_ciphertexts)
    totalCiphertextBits = 0
    for i in range(0, len(client_wise_norm_ciphertexts)):
        totalCiphertextBits += phantom.getCiphertextBits(client_wise_norm_ciphertexts[i])
    '\n    ciphertext [d0, d1, d2] : totalCiphertextBits * (2/3) when Sa -> Ss\n    '
    client_wise_norm, t_norm_dec = decrypt_norm_ciphertexts(context, args, scale, client_wise_norm_ciphertexts, sk1, sk1_remain, sk2, sk2_remain, step, is_version2, log_path)
    with open(log_path, 'a') as f:
        f.write(f'Sa -> Ss norm ciphertexts d1, d2 : {totalCiphertextBits * (2 / 3) / (8 * 1024.0 * 1024.0)}MB\n')
    del client_wise_norm_ciphertexts
    return (client_wise_norm, t_norm_he, t_norm_dec)

def Norm_Phase(context, args, scale, ciphertexts, modulus_len, step, sk1, sk1_remain, sk2, sk2_remain, is_version2, log_path, inverse_ciphertexts=None):
    client_wise_norm, t_norm_he, t_norm_dec = client_wise_norm_phase(context, args, scale, ciphertexts, modulus_len, step, sk1, sk1_remain, sk2, sk2_remain, is_version2, log_path, inverse_ciphertexts=inverse_ciphertexts)
    first_idxs, median = norm_check_using_modified_z_score(client_wise_norm, STD_THRESHOLD)
    return (client_wise_norm, median, first_idxs, t_norm_he, t_norm_dec)

def Similarity_ciphertexts(context, args, ciphertexts, global_update_plaintexts, modulus_len):
    start_sim_he = time.time()
    client_wise_similarity_ciphertexts = []
    for id in range(0, args.num_clients):
        client_ciphertexts = ciphertexts[id]
        client_similarity_ciphertext = multiply_and_sum_ciphertexts_similarity(context, client_ciphertexts, global_update_plaintexts, modulus_len)
        client_wise_similarity_ciphertexts.append(client_similarity_ciphertext)
    phantom.cudaStreamSychronize()
    t_sim_he = time.time() - start_sim_he
    return (client_wise_similarity_ciphertexts, t_sim_he)

def decrypt_similarity_ciphertexts(context, args, scale, client_wise_similarity_ciphertexts, sk1, sk2, client_wise_norm, is_version2, log_path):
    start_sim_dec = time.time()
    client_wise_similarity = []
    partialBits = 0
    for id in range(0, args.num_clients):
        client_norm = client_wise_norm[id]
        client_similarity_ciphertext = client_wise_similarity_ciphertexts[id]
        denominator = client_norm + eps_sim
        client_similarity, bits = similarity_check(context, scale, denominator, client_similarity_ciphertext, sk1, sk2, True, False, is_version2)
        partialBits += bits
        client_wise_similarity.append(client_similarity)
    t_sim_dec = time.time() - start_sim_dec
    return (client_wise_similarity, t_sim_dec)

def client_wise_similarity_phase(context, args, scale, ciphertexts, global_update_plaintexts, modulus_len, step, sk1, sk2, client_wise_norm, is_version2, log_path):
    client_wise_similarity_ciphertexts, t_sim_he = Similarity_ciphertexts(context, args, ciphertexts, global_update_plaintexts, modulus_len)
    client_wise_similarity, t_sim_dec = decrypt_similarity_ciphertexts(context, args, scale, client_wise_similarity_ciphertexts, sk1, sk2, client_wise_norm, is_version2, log_path)
    del client_wise_similarity_ciphertexts
    return (client_wise_similarity, t_sim_he, t_sim_dec)

def Similarity_Phase(context, args, scale, ciphertexts, global_update_plaintexts, modulus_len, step, sk1, sk2, client_wise_norm, is_version2, first_idxs, log_path):
    client_wise_similarity, t_sim_he, t_sim_dec = client_wise_similarity_phase(context, args, scale, ciphertexts, global_update_plaintexts, modulus_len, step, sk1, sk2, client_wise_norm, is_version2, log_path)
    last_survivor, median = similarity_check_using_modified_z_score(client_wise_similarity, first_idxs, STD_THRESHOLD)
    with open(log_path, 'a') as log:
        log.write(f'[Intersection] MAD=3.5 STD={STD_THRESHOLD:.10g} norm_ids={first_idxs} final_ids={last_survivor}\n')
    return (last_survivor, median, t_sim_he, t_sim_dec)
