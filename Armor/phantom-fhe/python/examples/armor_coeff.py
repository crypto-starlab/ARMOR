import pyPhantom as phantom
import numpy as np
import random
import time

log_n = 14
n= 2 ** log_n
modulus_chain = [62, 52, 52, 52]
size_P = 0
galois_steps = [0]
scale = 2.0 ** 88

def set_context():
    params = phantom.params(phantom.scheme_type.ckks) 
    params.set_poly_modulus_degree(n)
    params.set_special_modulus_size(size_P)
    params.set_galois_elts(phantom.get_elts_from_steps(galois_steps, n))
    flag = None
    
    params.set_coeff_modulus(phantom.create_coeff_modulus(n, modulus_chain))
    flag = False
    
    context = phantom.context(params)
    return context, flag
    

def set_key_setting(context, size_P):
    sk = phantom.secret_key(context)
    mult_key = sk.armor_gen_keys_single(context)

    if size_P > 0:
        relin_key = sk.gen_relinkey(context)
        rot_key = sk.create_galois_keys(context)
    else:
        relin_key = None
        rot_key = None
    pk = sk.gen_publickey(context)
    encoder = phantom.ckks_encoder(context)
    return sk, mult_key, relin_key, rot_key, pk, encoder

def encrypt_list(context, encoder, public_key, values, scale, size_P):
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {len(values)}")
    pt = encoder.armor_coefficient_encode(
        context, np.asarray(values, dtype=np.float64), scale, size_P
    )
    ciphertext = public_key.armor_encrypt_asymmetric(context, pt)
    phantom.cudaStreamSychronize()
    return ciphertext


def Ours(context, encoder, ciphertext, sk, mult_key, iter):
    decode_SOTA_time = 0.0
    decode_OUR_time = 0.0
    for i in range(0, iter):
        conjugated = phantom.armor_rotate_without_key_switching(
            context, ciphertext, galois_steps[0]
        )
        res = phantom.armor_multiply(context, ciphertext, conjugated)

        res = phantom.armor_to_intt(context, res)
        res = phantom.armor_rescale_to(context, res, len(modulus_chain) - 2)
        res = phantom.armor_to_ntt(context, res)
        phantom.cudaStreamSychronize()
        '''
        start dec
        '''
        dec = sk.armor_decrypt_cipher(context, res, mult_key, galois_steps[0])

        start = time.perf_counter()
        contants = phantom.armor_extract_constant(context, dec) 
        result = phantom.armor_decode_constant_single(context, contants, scale, 1.0)
        decode_OUR_time += time.perf_counter() - start

        print(result)

        phantom.cudaStreamSychronize()
        dec_pt = sk.armor_decrypt(context, res, mult_key, galois_steps[0])

        start = time.perf_counter()
        result_vec = encoder.armor_coefficient_decode_and_mult(context, dec_pt, 1.0)
        decode_SOTA_time += time.perf_counter() - start
        phantom.cudaStreamSychronize()
        #print(result_vec[0])
        print(result_vec[0] - result < 0.1)


    return decode_SOTA_time, decode_OUR_time


def main():
    values = [1.0 for _ in range(n)]
    context, _ = set_context()
    sk, mult_key, relin_key, rot_key, pk, encoder = set_key_setting(context, size_P)
    ciphertext = encrypt_list(context, encoder, pk, values, scale, size_P)
    iterations = 1
    SOTA_time, OUR_time = Ours(context, encoder, ciphertext, sk, mult_key, iterations)
    print(f"{'Ours':8s} | SOTA : {SOTA_time*1000:.3f} ms | OURS : {OUR_time*1000:.3f} ms")


if __name__ == "__main__":
    main()
