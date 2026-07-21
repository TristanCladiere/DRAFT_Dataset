import random
import torch
import pickle
from math import ceil


def get_k_splits(data_path, k):
    with open(f"{data_path}/data.pkl", "rb") as f:
        data = pickle.load(f)  # dict: key=job_id, val=DataFrame with cand_ids, labels, and sensitive attributes (gender, 0=female, 1=male)

    data = list(data.items())
    random.Random(42).shuffle(data)

    splits = []
    a, b = divmod(len(data), k)
    start = 0
    for i in range(k):
        end = start + a + (1 if i < b else 0)
        splits.append(dict(data[start:end]))
        start = end

    return splits


def create_k_cross_dataset(data_path, version, k_idx, k, for_cos_sim=False):
    splits = get_k_splits(data_path, k)

    jobs_embeddings = torch.load(f"{data_path}/jobs_embeddings_{version}.pt", map_location='cpu')
    candidates_embeddings = torch.load(f"{data_path}/candidates_embeddings_{version}.pt", map_location='cpu')

    train_data = dict()
    test_data = dict()
    for i, k_rank in enumerate(splits):
        if i == k_idx:
            test_data.update(k_rank)
        else:
            train_data.update(k_rank)
    nb_train_jobs = len(train_data)

    if not for_cos_sim:
        nb_train_jobs = ceil(0.9 * nb_train_jobs)  # 10% will be used to create a validation set

    pairs = {"Train": [], "Val": [], "Test": []}
    labels = {"Train": [], "Val": [], "Test": []}
    sas = {"Train": [], "Val": [], "Test": []}
    for i, (j_id, df) in enumerate(train_data.items()):
        set_name = "Train" if i < nb_train_jobs else "Val"
        for c_id, label, sa in df.values:
            j_emb = jobs_embeddings[j_id]
            c_emb = candidates_embeddings[c_id]
            p = torch.cat((j_emb, c_emb, torch.abs(j_emb-c_emb)), dim=0)
            pairs[set_name].append(p)
            labels[set_name].append(torch.tensor(1) if label > 0 else torch.tensor(0)) # relevant (1) and irrelevant (0) instead of 0, 1 and 2
            sas[set_name].append(sa)

    for j_id, df in test_data.items():
        for c_id, label, sa in df.values:
            j_emb = jobs_embeddings[j_id]
            c_emb = candidates_embeddings[c_id]
            p = torch.cat((j_emb, c_emb, torch.abs(j_emb-c_emb)), dim=0)
            pairs["Test"].append(p)
            labels["Test"].append(1 if label > 0 else 0) # relevant (1) and irrelevant (0) instead of 0, 1 and 2
            sas["Test"].append(sa)

    pairs["Test"] = torch.stack(pairs["Test"]).to(torch.float32)

    return test_data, pairs, labels, sas
