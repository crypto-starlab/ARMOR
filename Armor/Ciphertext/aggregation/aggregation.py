import pyPhantom as phantom
import time

def aggregation_sum(context, ciphertexts):
    start_agg_he = time.time()
    global_updates = []
    num_of_clients = len(ciphertexts)
    num_of_ciphertexts = len(ciphertexts[0])
    for i in range (0, num_of_ciphertexts):
        ct1 = ciphertexts[0][i]
        for j in range(1, num_of_clients):
            ct1 = phantom.armor_add(context, ct1, ciphertexts[j][i])
        global_updates.append(ct1)
    phantom.cudaStreamSychronize()
    t_agg_he = time.time() - start_agg_he
    return global_updates, t_agg_he



def aggregation_sum_last_survivor(context, ciphertexts, last_survivor):
    start_agg_he = time.time()
    global_updates = []
    ct_len = len(ciphertexts[0])
    for i in range(len(last_survivor)):
        idx = last_survivor[i]
        for j in range(ct_len):
            if i == 0:
                global_updates.append(ciphertexts[idx][j])
            else:
                global_updates[j] = phantom.armor_add(context, global_updates[j], ciphertexts[idx][j])

    phantom.cudaStreamSychronize()
    t_agg_he = time.time() - start_agg_he
    return global_updates, t_agg_he
