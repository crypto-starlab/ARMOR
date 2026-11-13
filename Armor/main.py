import traceback
import ctypes
import sys
from pathlib import Path
import random
import numpy as np
import torch
import copy
import os
import options
import dataset
import sampling
import train
import test
import gc
from Plaintext import utils as plain_utils
from Attacks.poisonedfl import PoisonedFL
from supplementary_jobs import experiment_output
from log_paths import accuracy_log_path
from Plaintext.intersection import intersection_armor
_he_import_error = None
try:
    _native_dir = Path(__file__).resolve().parent / 'phantom-fhe' / 'build' / 'lib'
    if not (_native_dir / 'libPhantom.so').exists() or not any(_native_dir.glob('pyPhantom*.so')):
        raise ImportError('Build the local native module: cmake --build phantom-fhe/build --target pyPhantom -j 8')
    ctypes.CDLL(str(_native_dir / 'libPhantom.so'), mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(_native_dir))
    import pyPhantom as phantom
    if Path(phantom.__file__).resolve().parent != _native_dir.resolve():
        raise ImportError(f'Expected local pyPhantom in {_native_dir}, loaded {phantom.__file__}')
    if not hasattr(phantom, 'getCiphertextBits'):
        raise ImportError(f'Rebuild local pyPhantom: missing getCiphertextBits in {phantom.__file__}')
    from Ciphertext.encrypt_decrypt import encrypt, decrypt, params
    from Ciphertext import utils
    from Ciphertext.aggregation import aggregation
except (ImportError, OSError) as error:
    _he_import_error = error

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_malicious_clients(args):
    max_num = max(int(args.attack_ratio * args.num_clients), 1)
    tmp_idx = [i for i in range(args.num_clients)]
    malicious_idxs = random.sample(tmp_idx, max_num)
    return malicious_idxs

def print_time(times):
    t_train, t_enc, t_norm_he, t_norm_dec, t_sim_he, t_sim_dec, t_agg_he, t_agg_dec = times
    print('===================================================')
    print(f'Train Time         : {t_train:.4f}')
    print(f'Encryption Time    : {t_enc:.4f}')
    print(f'Norm (HE)          : {t_norm_he:.4f}')
    print(f'Norm Decryption    : {t_norm_dec:.4f}')
    print(f'Similarity (HE)    : {t_sim_he:.4f}')
    print(f'Similarity Decrypt : {t_sim_dec:.4f}')
    print(f'Aggregation (HE)   : {t_agg_he:.4f}')
    print(f'Aggregation Decrypt: {t_agg_dec:.4f}')
    print('===================================================')

def run_experiment(args):
    """
    default setting
    """
    if args.cipher and _he_import_error is not None:
        raise RuntimeError('Cipher mode requires the matching pyPhantom build') from _he_import_error
    log_method = 'vanilla_armor' if args.method == 'armor' and not args.cipher else args.method
    args.log_path = args.log_path or str(Path(__file__).resolve().parent / accuracy_log_path(args.dataset, log_method, args.attack, args.attack_ratio).name)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    norm_protocol = getattr(args, 'norm_protocol', 'armor')
    protocol_suffix = ''
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_seed(getattr(args, 'seed', args.seed))
    num_clients_eval = args.num_clients if args.num_clients > 0 else 1
    dataset_train, dataset_test, dataset_val = dataset.load_data(args)
    dict_users = sampling.noniid(dataset_train, args)
    net_glob = dataset.init_model(args).to(args.device)
    orig_len = encrypt.flatten_weight(net_glob).numel() if args.cipher else 0
    malicious_idxs = set_malicious_clients(args) if args.attack_ratio > 0 else []
    if args.attack_ratio == 0.0:
        args.attack = 'No_Attack'
    if str(args.attack).lower() == 'poisonedfl':
        args._poisonedfl_state = PoisonedFL(net_glob, args)
        print(f'[PoisonedFL CONFIG] c={args._poisonedfl_state.scale:g}, window={args._poisonedfl_state.window}, malicious_clients={len(malicious_idxs)}, bootstrap_round={args.attack_time + 1}, first_attack_round={args.attack_time + 2}')
    '\n    parameter setting\n    '
    global_update = None
    if args.cipher:
        context, scale, step, modulus_len, is_version2 = params.set_context(args)
        sk1, sk2, sk1_remain, sk2_remain, public_key, encoder = params.set_key_setting(context, norm_protocol)
        print(f'[CONFIG] dataset={args.dataset}, method={args.method}, attack={args.attack}, attack_ratio={args.attack_ratio}, ring_dim={args.ringdim} ')
        args.log_path = args.log_path or f'./{args.dataset}_{args.method}{protocol_suffix}_{args.attack}_{args.attack_ratio}_log.txt'
        with open(args.log_path, 'w') as f:
            f.write(f'Experiment Setup - Dataset: {args.dataset} | Method: {args.method} | Attack: {args.attack} | Ratio: {args.attack_ratio} | Ring Dim: {args.ringdim}\n')
            f.write('=' * 60 + '\n')
    else:
        print(f'[CONFIG] dataset={args.dataset}, method={args.method}, attack={args.attack}, attack_ratio={args.attack_ratio}')
        log_method = 'vanilla_armor' if args.method == 'armor' else args.method
        args.log_path = getattr(args, 'log_path', None) or f'./{args.dataset}_{log_method}_{args.attack}_{args.attack_ratio}_log.txt'
    for iter in range(args.global_ep):
        t_train, t_norm_he, t_norm_dec, t_sim_he, t_sim_dec, t_agg_he, t_agg_dec = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        selected_clients = max(int(args.num_clients), 1)
        idxs_users = random.sample(range(args.num_clients), selected_clients)
        update_list, t_train = train.update_clients(idxs_users=idxs_users, dataset_train=dataset_train, dict_users=dict_users, args=args, iter=iter, malicious_idxs=malicious_idxs, net_glob=net_glob, global_update=global_update)
        ciphertexts = []
        inverse_ciphertexts = None
        global_update_plaintexts = None
        aggregate_len = 0
        if args.cipher:
            ciphertexts, t_enc = encrypt.encrypt_updates(context, encoder, update_list, public_key, scale, args)
            ciphertextBits = 0
            for i in range(0, len(ciphertexts[0])):
                ciphertextBits += phantom.getCiphertextBits(ciphertexts[0][i])
                if inverse_ciphertexts is not None:
                    ciphertextBits += phantom.getCiphertextBits(inverse_ciphertexts[0][i])
            with open(args.log_path, 'a') as f:
                f.write('=====================Encrypt Bits==========================\n')
                f.write(f'Ci->server1 : {ciphertextBits / (8 * 1024.0 * 1024.0)} MB\n')
                f.write('============================================================\n\n')
            if iter != 0:
                global_update_chunks = utils.global_update_linear(global_update, args)
                global_update_plaintexts = encrypt.encode_global_updates_conjugated(encoder, context, global_update_chunks, scale)
            if iter == 0 or args.method == 'fedavg':
                aggregate_len = args.num_clients
                encrypted_global_update, t_agg_he = aggregation.aggregation_sum(context, ciphertexts)
            elif iter != 0 and args.method == 'armor':
                with open(args.log_path, 'a') as f:
                    f.write('=====================Norm communication Bits==========================\n')
                client_wise_norm, _, first_survivors, t_norm_he, t_norm_dec = utils.Norm_Phase(context, args, scale, ciphertexts, modulus_len, step, sk1, sk1_remain, sk2, sk2_remain, is_version2, log_path=args.log_path, inverse_ciphertexts=inverse_ciphertexts)
                with open(args.log_path, 'a') as f:
                    f.write('======================================================================\n\n')
                with open(args.log_path, 'a') as f:
                    f.write('=====================Similarity communication Bits==========================\n')
                final_survivors, sim_variant, t_sim_he, t_sim_dec = utils.Similarity_Phase(context, args, scale, ciphertexts, global_update_plaintexts, modulus_len, step, sk1, sk2, client_wise_norm, is_version2, first_survivors, log_path=args.log_path)
                with open(args.log_path, 'a') as f:
                    f.write('======================================================================\n\n')
                aggregate_len = len(final_survivors)
                encrypted_global_update, t_agg_he = aggregation.aggregation_sum_last_survivor(context, ciphertexts, final_survivors)
            with open(args.log_path, 'a') as f:
                f.write('=====================aggregation communication Bits==========================\n')
            aggregate_update, t_agg_dec = decrypt.decrypt_global_update(context, encoder, encrypted_global_update, sk1, sk2, aggregate_len, args.log_path)
            with open(args.log_path, 'a') as f:
                f.write('=============================================================================\n\n')
            global_update = decrypt.update_global_model(net_glob, aggregate_update, orig_len, args)
        else:
            t_enc = 0.0
            c = max(int(args.attack_ratio * args.num_clients), 1)
            try:
                if args.method == 'fedavg':
                    if iter == 0:
                        aggregate_update, _, _ = plain_utils.FedAvg(update_list)
                    else:
                        aggregate_update, _, _ = plain_utils.FedAvg(update_list, global_update)
                elif args.method == 'krum':
                    res = plain_utils.krum(update_list, c, args)
                    aggregate_update = res[0] if isinstance(res, tuple) else res
                elif args.method == 'multi_krum':
                    res = plain_utils.multi_krum(update_list, c, args)
                    aggregate_update = res[0] if isinstance(res, tuple) else res
                elif args.method == 'median':
                    aggregate_update = plain_utils.median(update_list)
                elif args.method == 'trimmed_mean':
                    trim_count = args.trim_count if args.trim_count is not None else len(malicious_idxs)
                    aggregate_update = plain_utils.trimmed_mean(update_list, trim_count)
                elif args.method == 'cosine':
                    if global_update is not None:
                        aggregate_update, _, _ = plain_utils.cosine_defense(update_list, global_update, args)
                    else:
                        aggregate_update, _, _ = plain_utils.FedAvg(update_list)
                elif iter == 0:
                    aggregate_update, _, _ = plain_utils.FedAvg(update_list)
                else:
                    aggregate_update, _, _ = plain_utils.Armor(update_list, global_update, args, client_ids=idxs_users, malicious_idxs=malicious_idxs, round_idx=iter)
            except Exception as e:
                print(f'\n[CRITICAL ERROR] {iter + 1}round {args.method} method has error!')
                traceback.print_exc()
                raise
            w_glob = net_glob.state_dict()
            for k in w_glob.keys():
                w_glob[k] += aggregate_update[k].to(args.device).to(device=args.device, dtype=w_glob[k].dtype)
            net_glob.load_state_dict(w_glob)
            global_update = copy.deepcopy(aggregate_update)
        test_acc, test_loss = test.test_img(net_glob.to(args.device), dataset_test, iter, args)
        with open(args.log_path, 'a') as f:
            f.write(f'{iter + 1} Round acc : {test_acc:.4f} | Loss: {test_loss:.4f}\n')
            if args.cipher:
                f.write(f'{iter + 1} t_enc: {t_enc:.8f}\n')
                f.write(f'{iter + 1} t_norm_he, t_norm_dec: {t_norm_he:.8f}, {t_norm_dec:.8f}\n')
                f.write(f'{iter + 1} t_sim_he, t_sim_dec: {t_sim_he:.8f}, {t_sim_dec:.8f}\n')
                f.write(f'{iter + 1} t_agg_he, t_agg_dec: {t_agg_he:.8f}, {t_agg_dec:.8f}\n\n')
        times = (t_train, t_enc, t_norm_he, t_norm_dec, t_sim_he, t_sim_dec, t_agg_he, t_agg_dec)
        print_time(times)
        if args.cipher:
            del ciphertexts
            del inverse_ciphertexts
            del encrypted_global_update
            if global_update_plaintexts is not None:
                del global_update_plaintexts
        del aggregate_update
        gc.collect()
        torch.cuda.empty_cache()
def main(argv=None):
    args = options.args_parser(argv)
    if args.armor_filter == 'intersection':
        if args.method != 'armor' or args.cipher:
            raise ValueError('Intersection option requires plaintext ARMOR')
        plain_utils.Armor = intersection_armor
        print('[CONFIG] MAD=3.5 AND population STD=1.779; norm AND similarity', flush=True)
    with experiment_output(args) as log_file:
        if log_file is not None:
            run_experiment(args)


if __name__ == '__main__':
    main()

