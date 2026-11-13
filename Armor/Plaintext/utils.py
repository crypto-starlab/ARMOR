import copy
import random
import torch
import numpy as np
from armor_filter_config import STD_THRESHOLD


def _defense_inputs(updates):
    """Validate independent state-dict deltas; aggregate only floating buffers.

    Integer buffers (e.g. BatchNorm counters) receive zero deltas. Inputs are
    never modified. Reject non-finite clients before computing statistics.
    """
    if not updates:
        raise ValueError('Defense requires at least one client update')
    reference = updates[0]
    keys = [k for k, v in reference.items() if v.is_floating_point()]
    if not keys:
        raise ValueError('Defense requires floating-point updates')
    valid = []
    for update in updates:
        if update.keys() != reference.keys() or any(
                update[k].shape != v.shape or update[k].dtype != v.dtype or
                update[k].device != v.device for k, v in reference.items()):
            raise ValueError('Client update keys, shapes, dtypes and devices must match')
        if all(torch.isfinite(update[k]).all().item() for k in keys):
            valid.append(update)
    if not valid:
        raise ValueError('No finite client updates remain')
    result = type(reference)((k, torch.zeros_like(v)) for k, v in reference.items())
    return valid, keys, result


@torch.no_grad()
def median(update_locals):
    """Coordinate-wise median, averaging the middle pair for even counts."""
    updates, keys, result = _defense_inputs(update_locals)
    n = len(updates)
    for key in keys:
        values = torch.stack([u[key] for u in updates]).sort(dim=0).values
        result[key] = values[n // 2].clone() if n % 2 else (
            values[n // 2 - 1] * .5 + values[n // 2] * .5)
    return result


@torch.no_grad()
def trimmed_mean(update_locals, trim_count):
    """Remove trim_count smallest AND largest values at each coordinate."""
    updates, keys, result = _defense_inputs(update_locals)
    n = len(updates)
    if isinstance(trim_count, bool) or not isinstance(trim_count, int) or not 0 <= 2 * trim_count < n:
        raise ValueError('trim_count must be an integer >= 0 with 2 * trim_count < finite clients')
    for key in keys:
        values = torch.stack([u[key] for u in updates]).sort(dim=0).values
        result[key] = values[trim_count:n - trim_count].mean(dim=0)
    return result


def format_tensor(data, device = 'cuda'):
    if isinstance(data, torch.Tensor):
        return data.flatten().to(device)
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], torch.Tensor):
        return torch.cat([d.flatten() for d in data]).to(device)
    else:
        return torch.tensor(data, device=device).flatten()

def get_norm_chunk(global_layers, device = 'cuda'):
    g_tensor = format_tensor(global_layers, device)
    g_norm = torch.norm(g_tensor, p=2)

    return g_norm.item()
def calculate_inner_product_torch(global_layers, client_plaintexts, device='cuda'):
    g_tensor = format_tensor(global_layers, device)
    c_tensor = format_tensor(client_plaintexts, device)
    
    inner_product = torch.sum(g_tensor * c_tensor)
    
    return inner_product.item()

def FedAvg(w_locals, global_update=None):
    if not w_locals:
        return None, 0.0, 0.0

    w_avg = copy.deepcopy(w_locals[0])
    num_clients = len(w_locals)
    
    base_keys = [k for k, v in sorted(w_avg.items()) if v.is_floating_point()]

    for key in base_keys:
        for i in range(1, num_clients):
            if key in w_locals[i]:
                w_avg[key] += w_locals[i][key]
        w_avg[key] = torch.div(w_avg[key], float(num_clients))

    if global_update is not None:
        ref_v = torch.cat([global_update[k].flatten() for k in base_keys if k in global_update])
    else:
        ref_v = torch.cat([w_avg[k].flatten() for k in base_keys])

    norm_ref = torch.norm(ref_v, p=2) + 1e-12

    flat_updates = []
    norms = []
    sims = []

    for w in w_locals:
        tensors = []
        for k in base_keys:
            if k in w and w[k].numel() == w_avg[k].numel():
                tensors.append(w[k].flatten())
            else:
                tensors.append(torch.zeros_like(w_avg[k]).flatten())
                
        flat_v = torch.cat(tensors)
        flat_updates.append(flat_v)
        
        n = torch.norm(flat_v, p=2).item()
        norms.append(n)
        
        sim = torch.sum(flat_v * ref_v) / ((n + 1e-12) * norm_ref)
        sims.append(sim.item())

    median_norm = float(np.median(norms))
    median_sim = float(np.median(sims))

    return w_avg, median_norm, median_sim


def Byzantine_Robust_Fed_Avg(update_list, last_survivor, local_idxs):
    if not update_list or not last_survivor or not local_idxs:
        return None

    final_update_list = []
    for i in range(0, len(last_survivor)):
        final_update_list.append(update_list[last_survivor[i]])
    
    return FedAvg(final_update_list)
    
def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)

def unflatten_vector(vector, reference_dict):
    new_dict = copy.deepcopy(reference_dict)
    pointer = 0
    for k, v in reference_dict.items():
        numel = v.numel()
        tensor_slice = vector[pointer : pointer + numel].view_as(v)
        new_dict[k] = tensor_slice.to(dtype=v.dtype, device=v.device)
        pointer += numel
    return new_dict

def flatten_vector(global_update, args):
    flattened_tensors = []
    for tensor in global_update.values():
        flattened_tensors.append(tensor.reshape(-1).to(args.device))

    vector = torch.cat(flattened_tensors)
    return vector

def flatten_vectors(w_locals, args):
    vectors = []
    
    for v in w_locals:
        flattened_tensors = []
        for name in v:
            flattened_tensors.append(v[name].reshape(-1).to(args.device))
        
        vectors.append(torch.cat(flattened_tensors))

    return vectors


def distance_matrix(update_locals, args):
    vectors = flatten_vectors(update_locals, args)
    distance = torch.zeros([len(vectors), len(vectors)]).to(args.device)
    
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
                
    return distance


def krum(update_locals, c, args):

    n = len(update_locals) - c
   
    distance = distance_matrix(update_locals, args)
    sorted_idx = distance.sum(dim=0).argsort()[: n]
    chosen_idx = int(sorted_idx[0])
    

    print(f"We select {sorted_idx[0]} as our aggerated client.")
    return copy.deepcopy(update_locals[chosen_idx]), chosen_idx

def multi_krum(update_locals, c, args, multi_k=True, m=None):

    N = len(update_locals)
    
    dist_matrix = distance_matrix(update_locals, args)
    
    remaining_indices = list(range(N))
    chosen_indices = []
    
    while len(remaining_indices) > 2 * c + 2:
        rem_tensor = torch.tensor(remaining_indices, dtype=torch.long, device=args.device)
        sub_dist = dist_matrix[rem_tensor][:, rem_tensor]
        
        sorted_dist, _ = torch.sort(sub_dist, dim=1)
        
        k = len(remaining_indices) - 2 - c
        if k <= 0:
            break
            
        scores = torch.sum(sorted_dist[:, :k], dim=1)
        
        min_idx = torch.argmin(scores).item()
        chosen_client = remaining_indices[min_idx]
        
        chosen_indices.append(chosen_client)
        remaining_indices.pop(min_idx)
        
        if not multi_k:
            break
            
        if m is not None and len(chosen_indices) >= m:
            break

    print(f"We selected {len(chosen_indices)} clients as our aggregated clients: {chosen_indices}")
    
    update_avg = copy.deepcopy(update_locals[chosen_indices[0]])
    for key in update_avg.keys():
        for i in range(1, len(chosen_indices)):
            update_avg[key] += update_locals[chosen_indices[i]][key]
        
        update_avg[key] = torch.div(update_avg[key], float(len(chosen_indices)))
        
    return update_avg, chosen_indices


def cosine_defense(update_locals, global_update, args):

    global_update_vector = flatten_vector(global_update, args)
    vectors = flatten_vectors(update_locals, args)
    similarity_list = []
    for i in range(0, len(vectors)):
        client_update_vec = vectors[i]
        sim = torch.cosine_similarity(global_update_vector, client_update_vec, dim = 0)
        similarity_list.append(abs(sim.item()))
    
    cs_tensor = torch.tensor(similarity_list, dtype=torch.float32, device=args.device)
    max_val = torch.max(cs_tensor)
    min_val = torch.min(cs_tensor)
    
    if max_val - min_val == 0:
        normalized_cs = torch.zeros_like(cs_tensor) 
    else:
        normalized_cs = (cs_tensor - min_val) / (max_val - min_val)
        
    threshold = torch.mean(normalized_cs).item()

    benign_update = []
    for i in range(len(normalized_cs)):
        if normalized_cs[i].item() < threshold:
            benign_update.append(update_locals[i])

    
    if not benign_update:
        return FedAvg(update_locals)
    
    return FedAvg(benign_update)




def norm_check_using_modified_z_score(norm_lists, z_thresh=STD_THRESHOLD):
    norms = np.array([x for x in norm_lists], dtype=np.float64)
    median = np.median(norms)
    std = np.std(norms, ddof=0)

    if std == 0:
        print("STD == 0 (all values are identical)")
        return 0

    robust_z_scores = np.abs(norms - median) / std

    passed_idx = []

    for i in range(0, len(robust_z_scores)):
        z = robust_z_scores[i]
        if z <= z_thresh:
            passed_idx.append(i)
            
    return passed_idx


def similarity_check_using_modified_z_score(client_wise_similarity, first_remain, z_thresh=STD_THRESHOLD):
    similaritys = torch.tensor([x for x in client_wise_similarity], dtype=torch.float64)
    median = torch.median(similaritys)
    std = torch.std(similaritys, unbiased=False)

    if std.item() == 0:
        print("STD == 0 (all values are identical)")
        return 0

    robust_z_scores = torch.abs(similaritys - median) / std
    
    passed_idxs = []

    for i in range(len(robust_z_scores)):
        z = robust_z_scores[i]
        if z.item() <= z_thresh:
            passed_idxs.append(i)

    final_idxs = [idx for idx in first_remain if idx in passed_idxs]
    
    return final_idxs



def dividing_by_chunk(vector, args):
    padding = (-vector.numel()) % args.ringdim
    if padding:
        vector = torch.nn.functional.pad(vector, (0, padding))
    return list(vector.view(-1, args.ringdim))


def Armor(update_locals, global_update, args, client_ids=None, malicious_idxs=None, round_idx=None):
    global_update_vector = flatten_vector(global_update, args)
    vectors = flatten_vectors(update_locals, args)
    norms = [torch.norm(vector, p=2).item() for vector in vectors]

    norm_median = float(np.median(norms))

    passed_norm_idxs = norm_check_using_modified_z_score(norms)

    if isinstance(passed_norm_idxs, int) and passed_norm_idxs == 0:
        passed_norm_idxs = list(range(len(update_locals)))


    similarities = []

    for subvec in vectors:
        sim = torch.cosine_similarity(global_update_vector, subvec, dim=0)
        similarities.append(sim.item())

    similarity_median = float(np.median(similarities))

    log_path = getattr(args, 'log_path', None)
    if log_path and round_idx is not None and round_idx >= 50:
        logged_client_ids = client_ids if client_ids is not None else range(len(norms))
        with open(log_path, 'a') as f:
            f.write(f"[Armor] Round {round_idx + 1} client norms and similarities (full vectors)\n")
            f.write(f"Norm median (all clients): {norm_median:.10g}\n")
            f.write(f"Similarity median (all clients): {similarity_median:.10g}\n")
            for client_id, norm, similarity in zip(logged_client_ids, norms, similarities):
                f.write(f"Client {client_id}: norm={norm:.10g}, similarity={similarity:.10g}\n")

    final_passed_idxs = similarity_check_using_modified_z_score(client_wise_similarity=similarities, first_remain=passed_norm_idxs)

    if isinstance(final_passed_idxs, int) and final_passed_idxs == 0:
        final_passed_idxs = passed_norm_idxs

    if not final_passed_idxs:
        print(
            "[Armor Defense] Warning: No survivors left. "
            "Falling back to FedAvg with all clients."
        )
        aggregated_update, _, _ = FedAvg(update_locals)
    else:
        benign_updates = [update_locals[idx] for idx in final_passed_idxs]
        aggregated_update, _, _ = FedAvg(benign_updates)

    return aggregated_update, norm_median, similarity_median
