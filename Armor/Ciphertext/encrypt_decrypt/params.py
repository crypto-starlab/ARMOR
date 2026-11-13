import pyPhantom as phantom
modulus_chain_ver1 = [62, 53, 53, 53]
modulus_chain_ver2 = [62, 58, 58]
galois_steps = [0]
size_P = 0
scale_version1 = 2.0 ** 92
scale_version2 = 2.0 ** 74

def set_context_version1(args):
    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(args.ringdim)
    params.set_coeff_modulus(phantom.create_coeff_modulus(args.ringdim, modulus_chain_ver1))
    params.set_special_modulus_size(size_P)
    params.set_galois_elts(phantom.get_elts_from_steps(galois_steps, args.ringdim))
    context = phantom.context(params)
    return (context, scale_version1, galois_steps[0], len(modulus_chain_ver1), False)

def set_context_version2(args):
    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(args.ringdim)
    params.set_coeff_modulus(phantom.create_coeff_modulus(args.ringdim, modulus_chain_ver2))
    params.set_special_modulus_size(size_P)
    params.set_galois_elts(phantom.get_elts_from_steps(galois_steps, args.ringdim))
    context = phantom.context(params)
    return (context, scale_version2, galois_steps[0], len(modulus_chain_ver2), True)

def set_context(args):
    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(args.ringdim)
    params.set_special_modulus_size(size_P)
    params.set_galois_elts(phantom.get_elts_from_steps(galois_steps, args.ringdim))
    flag = None
    if args.dataset in ['CIFAR10', 'FMNIST']:
        params.set_coeff_modulus(phantom.create_coeff_modulus(args.ringdim, modulus_chain_ver1))
        modulus = modulus_chain_ver1
        scale = scale_version1
        flag = False
    else:
        params.set_coeff_modulus(phantom.create_coeff_modulus(args.ringdim, modulus_chain_ver2))
        modulus = modulus_chain_ver2
        scale = scale_version2
        flag = True
    context = phantom.context(params)
    return (context, scale, galois_steps[0], len(modulus), flag)

def set_key_setting(context, norm_protocol='armor'):
    sk = phantom.secret_key(context)
    [sk1, sk2, sk1_rand, sk2_remain] = sk.armor_gen_keys(context)
    pk = sk.gen_publickey(context)
    encoder = phantom.ckks_encoder(context)
    return (sk1, sk2, sk1_rand, sk2_remain, pk, encoder)
