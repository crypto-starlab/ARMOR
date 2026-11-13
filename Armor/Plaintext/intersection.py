"""Plaintext ARMOR with intersected MAD 3.5 and population STD 1.779 filters."""
import json
import numpy as np
import torch
from Plaintext import utils

from armor_filter_config import MAD_THRESHOLD, STD_THRESHOLD


def filter_sets(values, similarity=False, std_threshold=STD_THRESHOLD):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    # Nonfinite inputs cannot produce a valid distribution or survivors.
    if not finite.all():
        return set(), set()
    median = (np.sort(values)[(len(values) - 1) // 2] if similarity
              else np.median(values))
    deviations = np.abs(values - median)
    mad = (np.sort(deviations)[(len(values) - 1) // 2] if similarity
           else np.median(deviations))
    std = np.std(values, ddof=0)

    def passing(denominator, threshold, coefficient=1.0):
        if denominator == 0:
            return set(range(len(values)))
        return set(np.flatnonzero(coefficient * deviations / denominator <= threshold).tolist())

    return passing(mad, MAD_THRESHOLD, coefficient=0.6745), passing(std, std_threshold)


def intersection_armor(update_locals, global_update, args, client_ids=None,
                       malicious_idxs=None, round_idx=None):
    reference = utils.flatten_vector(global_update, args)
    vectors = utils.flatten_vectors(update_locals, args)
    norms = [torch.norm(vector, p=2).item() for vector in vectors]
    similarities = [torch.cosine_similarity(reference, vector, dim=0).item()
                    for vector in vectors]
    mad_norm, std_norm = filter_sets(norms)
    mad_sim, std_sim = filter_sets(similarities, similarity=True)
    mad_survivors = mad_norm & mad_sim
    std_survivors = std_norm & std_sim
    survivors = sorted(mad_survivors & std_survivors)
    ids = list(client_ids) if client_ids is not None else list(range(len(vectors)))
    def clients(indices):
        return [int(ids[index]) for index in sorted(indices)]
    record = dict(round=None if round_idx is None else round_idx + 1,
                  mad_threshold=MAD_THRESHOLD, std_threshold=STD_THRESHOLD,
                  mad_norm_ids=clients(mad_norm), std_norm_ids=clients(std_norm),
                  mad_similarity_ids=clients(mad_sim), std_similarity_ids=clients(std_sim),
                  mad_ids=clients(mad_survivors), std_ids=clients(std_survivors),
                  intersection_ids=clients(survivors),
                  mad_count=len(mad_survivors), std_count=len(std_survivors),
                  intersection_count=len(survivors),
                  empty_policy='skip_global_update')
    norm_median = float(np.median(norms))
    sim_median = float(np.median(similarities))
    if args.log_path:
        with open(args.log_path, 'a') as log:
            log.write('[Intersection] ' + json.dumps(record) + '\n')
            if round_idx is not None and round_idx >= 50:
                log.write(f'[Armor] Round {round_idx + 1} client norms and similarities (full vectors)\n')
                log.write(f'Norm median (all clients): {norm_median:.10g}\n')
                log.write(f'Similarity median (all clients): {sim_median:.10g}\n')
                for client, norm, sim in zip(ids, norms, similarities):
                    log.write(f'Client {int(client)}: norm={norm:.10g}, similarity={sim:.10g}\n')
    print(f'[Intersection] round={record["round"]} MAD={len(mad_survivors)} '
          f'STD={len(std_survivors)} both={len(survivors)}', flush=True)
    if survivors:
        update, _, _ = utils.FedAvg([update_locals[i] for i in survivors])
    else:
        # Never reintroduce rejected clients via the baseline FedAvg fallback.
        update = {key: torch.zeros_like(value) for key, value in update_locals[0].items()}
    return update, norm_median, sim_median

