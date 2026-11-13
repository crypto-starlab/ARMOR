import argparse
from Attacks.model_attacks import attack_name
from Attacks.data_attacks import is_label_flipping

def args_parser(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str.lower, default='armor', choices=['armor', 'fedavg', 'krum', 'multi_krum', 'cosine', 'median', 'trimmed_mean'], help='aggregation method')
    parser.add_argument('--armor_filter', choices=['default', 'intersection'], default='default', help='intersection: MAD 3.5 AND population STD 1.779')
    parser.add_argument('--trim_count', type=int, default=None, help='trimmed_mean: values removed from EACH tail; default is actual malicious client count')
    parser.add_argument('--global_ep', type=int, default=200, help='total number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=50, help='number of clients: K')
    parser.add_argument('--cipher', action='store_true', help='use phantom fhe')
    parser.add_argument('--norm_protocol', choices=['armor'], default='armor', help='ARMOR server automorphism')
    parser.add_argument('--local_ep', type=int, default=1, help='number of local epochs: E')
    parser.add_argument('--local_bs', type=int, default=32, help='local batch size: B')
    parser.add_argument('--bs', type=int, default=32, help='test batch size')
    parser.add_argument('--ds', type=int, default=20, help='dummy batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1 for CPU')
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 1)')
    parser.add_argument('--ringdim', type=int, default=16384, help='he ring dimension')
    parser.add_argument('--log_path', default=None, help='Distributed log file (supports {role})')
    parser.add_argument('--extract_partial_constant', action=argparse.BooleanOptionalAction, default=True, help='Extract partial constants; disable for full coefficient decoding')
    parser.add_argument('--attack_time', default=50, type=int, help='start attack time')
    parser.add_argument('--attack_ratio', default=0.0, type=float, help='fraction of compromised clients')
    parser.add_argument('--attack', type=str, default='No_Attack', help='attack (scaling, MPAF, ALIE, MIN_SUM, MIN_MAX, Sign_flipping, poisonedfl, label_flipping)')
    parser.add_argument('--label_flip_source', type=int, default=None, help='Optional source class; specify with target')
    parser.add_argument('--label_flip_target', type=int, default=None, help='Optional target class; default mapping is y -> C-1-y')
    parser.add_argument('--attack_reference', choices=['malicious', 'benign'], default='malicious', help='Statistics from attacker local deltas or omniscient benign deltas')
    parser.add_argument('--alie_z', type=float, default=None, help='ALIE z; default derives from selected client counts')
    parser.add_argument('--min_direction', choices=['std', 'sign', 'unit'], default='std')
    parser.add_argument('--min_lambda_max', type=float, default=10.0, help='Min-Sum/Max search upper bound')
    parser.add_argument('--min_tolerance', type=float, default=1e-05)
    parser.add_argument('--sign_flip_scale', type=float, default=1.0)
    parser.add_argument('--poisonedfl_scale', type=float, default=8.0, help='PoisonedFL initial c factor')
    parser.add_argument('--poisonedfl_window', type=int, default=50, help='PoisonedFL feedback window in attack rounds')
    parser.add_argument('--poisonedfl_decay', type=float, default=0.7, help='PoisonedFL c decay on unsuccessful feedback')
    parser.add_argument('--poisonedfl_min_scale', type=float, default=0.5, help='PoisonedFL c lower bound')
    parser.add_argument('--poisonedfl_pvalue', type=float, default=0.01, help='PoisonedFL binomial test significance')
    parser.add_argument('--keycenter_host', default='127.0.0.1')
    parser.add_argument('--keycenter_port', type=int, default=9100)
    parser.add_argument('--server1_host', default='127.0.0.1')
    parser.add_argument('--server1_port', type=int, default=9101)
    parser.add_argument('--server2_host', default='127.0.0.1')
    parser.add_argument('--server2_port', type=int, default=9102)
    parser.add_argument('--connect_timeout', type=float, default=300)
    args = parser.parse_args(argv)
    name = args.attack.lower().replace('-', '_')
    if name not in {'no_attack', 'scaling', 'mpaf', 'poisonedfl'} and attack_name(args.attack) is None and (not is_label_flipping(args.attack)):
        parser.error('Unsupported attack: ' + args.attack)
    if args.num_clients < 1 or args.local_bs < 1 or args.local_ep < 1 or (args.global_ep < 1):
        parser.error('Client count, batch size and epoch counts must be positive')
    if not 0 <= args.attack_ratio <= 1:
        parser.error('attack_ratio must be in [0,1]')
    if args.cipher and args.method not in {'armor', 'fedavg'}:
        parser.error('Cipher mode supports only armor and fedavg; use plaintext for this defense')
    if args.trim_count is not None and args.trim_count < 0:
        parser.error('trim_count must be nonnegative')
    if args.method == 'trimmed_mean':
        count = args.trim_count if args.trim_count is not None else max(int(args.attack_ratio * args.num_clients), 1) if args.attack_ratio > 0 else 0
        if 2 * count >= args.num_clients:
            parser.error('trimmed_mean requires 2 * trim_count < num_clients')
    if (args.label_flip_source is None) != (args.label_flip_target is None):
        parser.error('Specify both --label_flip_source and --label_flip_target')
    if args.label_flip_source is not None and (not 0 <= args.label_flip_source < args.num_classes or not 0 <= args.label_flip_target < args.num_classes or args.label_flip_source == args.label_flip_target):
        parser.error('Label flipping source and target must be distinct valid classes')
    return args
