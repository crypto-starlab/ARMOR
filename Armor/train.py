import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DatasetSplit
from collections import OrderedDict
from Attacks.data_attacks import is_label_flipping, flip_labels
import time
from Attacks.poisonedfl import PoisonedFL
from Attacks.model_attacks import apply_attacks

class ClientUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        num_classes = getattr(self.args, 'num_classes', 10)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, num_classes=num_classes), 
                                    batch_size=self.args.local_bs, 
                                    shuffle=True, 
                                    drop_last=True)

    def train(self, net, is_malicious=False, *, initial_state, timings=None):
        if is_malicious and str(self.args.attack).lower() == "mpaf":
            mpaf = OrderedDict()
            scale = 5.0
            for k, v in net.state_dict().items():
                mpaf[k] = v.clone().detach()
            for k in mpaf.keys():
                tmp = torch.zeros_like(mpaf[k], dtype=torch.float32).to(self.args.device)
                w_base = torch.randn_like(mpaf[k], dtype=torch.float32).to(self.args.device)
                tmp += (mpaf[k].to(self.args.device) - w_base) * scale
                mpaf[k].copy_(tmp)
            return mpaf

        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9)

        for _ in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if is_malicious and is_label_flipping(self.args.attack):
                    labels = flip_labels(labels, self.args.num_classes,
                                         getattr(self.args, 'label_flip_source', None),
                                         getattr(self.args, 'label_flip_target', None))
                
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels.reshape(-1))
                loss.backward()
                optimizer.step()

        if timings is not None:
            if torch.device(self.args.device).type == 'cuda':
                torch.cuda.synchronize(self.args.device)
            timings['training_end_s'] = time.perf_counter()
        update_dict = net.state_dict()
        with torch.no_grad():
            for k, value in update_dict.items():
                value.sub_(initial_state[k])

        if is_malicious:
            if self.args.attack == "scaling":
                scaling = 100.0
                for k in update_dict.keys():
                    if update_dict[k].is_floating_point():
                        update_dict[k] *= scaling

        return update_dict


def update_clients(idxs_users, dataset_train, dict_users,
                    args, iter, malicious_idxs, net_glob, global_update=None, timings=None):
    update_list = []
    malicious_positions = []
    train_device = torch.device(args.device)

    if train_device.type == 'cuda':
        torch.cuda.synchronize(train_device)
    start_train = time.perf_counter()
    initial_state = net_glob.state_dict()
    if str(args.attack).lower() == 'mpaf' and iter == args.attack_time:
        message = (f'[Attack] round={iter + 1} type=MPAF '
                   f'malicious={sum(client in malicious_idxs for client in idxs_users)} scale=5.0')
        print(message, flush=True)
        with open(args.log_path, 'a') as log:
            log.write(message + '\n')
    if timings is not None:
        timings.update(training_s=0.0, postprocess_s=0.0)

    def timestamp():
        if train_device.type == 'cuda':
            torch.cuda.synchronize(train_device)
        return time.perf_counter()

    postprocess_start = timestamp() if timings is not None else None

    poisoned_update = None
    if str(args.attack).lower() == 'poisonedfl' and iter >= args.attack_time \
            and any(client in malicious_idxs for client in idxs_users):
        if not hasattr(args, '_poisonedfl_state'):
            args._poisonedfl_state = PoisonedFL(net_glob, args)
        poisoned_update = args._poisonedfl_state.craft(net_glob)
        stats = args._poisonedfl_state.last_stats
        message = (f"[PoisonedFL] round={iter + 1} c={stats['scale']:.6g} "
                   f"history_norm={stats['history_norm']:.6g} "
                   f"attack_norm={stats['attack_norm']:.6g} "
                   f"aligned={stats['aligned']} threshold={stats['threshold']}")
        print(message)
        with open(args.log_path, 'a') as log:
            log.write(message + '\n')


    if timings is not None:
        timings['postprocess_s'] += timestamp() - postprocess_start
    for id in idxs_users:
        client_start = timestamp() if timings is not None else None
        is_malicious = (iter >= args.attack_time) and (id in malicious_idxs)
        if is_malicious and poisoned_update is not None:
            update_list.append(OrderedDict((k, v.detach().clone())
                                           for k, v in poisoned_update.items()))
            if timings is not None:
                timings['postprocess_s'] += timestamp() - client_start
            continue
        local = ClientUpdate(args=args, dataset=dataset_train, idxs=dict_users[id])
        local_timings = {} if timings is not None else None
        kwargs = dict(is_malicious=is_malicious, initial_state=initial_state)
        if timings is not None:
            kwargs['timings'] = local_timings
        local_update = local.train(net=copy.deepcopy(net_glob).to(args.device), **kwargs)
        if timings is not None:
            client_end = timestamp()
            
            training_end = local_timings.get('training_end_s', client_start)
            timings['training_s'] += training_end - client_start
            timings['postprocess_s'] += client_end - training_end
        if is_malicious:
            malicious_positions.append(len(update_list))
        update_list.append(local_update)
   
    postprocess_start = timestamp() if timings is not None else None
    stats = apply_attacks(update_list, malicious_positions, net_glob, args)
    if stats:
        message = f"[Attack] round={iter + 1} type={args.attack} malicious={len(malicious_positions)} {stats}"
        print(message)
        if getattr(args, 'log_path', None):
            with open(args.log_path, 'a') as log:
                log.write(message + '\n')
    if train_device.type == 'cuda':
        torch.cuda.synchronize(train_device)
    t_train = time.perf_counter() - start_train
    if timings is not None:
        timings['postprocess_s'] += time.perf_counter() - postprocess_start
        timings['whole_call_s'] = t_train
        t_train = timings['training_s']
   
    return update_list, t_train
