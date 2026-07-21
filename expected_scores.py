import math
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataset import get_k_splits
from evaluator import RCFEvaluator, compute_dcg_at_k, create_results_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='Data')
    parser.add_argument('--nb_draw', type=int, default=int(500))
    _args = parser.parse_args()
    return _args


def expected_random_acc_at_k(nb_cand, nb_rel, k):
    miss = math.perm(nb_cand-nb_rel, k) / math.perm(nb_cand, k)
    return 1.0 - miss


if __name__ == "__main__":
    args = parse_args()

    splits = get_k_splits(args.data_folder, k=1)
    split = splits[0]
    relevances = []
    for j_idx, df in split.items():
            relevances.extend(df.label.tolist())
    labels = (np.asarray(relevances) > 0).astype(int)

    # BEST POSSIBLE SCORES
    ####################################################################################################################
    evaluator = RCFEvaluator(ranking_data=split,
                             classif_data=labels.tolist(),
                             mrr_at_k=[3],
                             ndcg_at_k=[3],
                             hr_at_k=[3],
                             precision_recall_at_k=[3],
                             map_at_k=[3])
    gt_results, txt_job_rank, txt_job_fair = evaluator(relevances, labels.tolist())
    print(f"\n* BEST POSSIBLE SCORES *\n")
    print(txt_job_rank)
    print(f"{txt_job_fair}\n")
    print(create_results_table(gt_results))
    ####################################################################################################################

    # CLASSIFICATION
    ####################################################################################################################
    nb_pos = labels.sum()
    total = len(labels)
    q = nb_pos / total

    # Expected random scores
    e_acc = e_rec = 0.5
    e_pre = q
    e_f1 = q / (q + 0.5)

    # Estimation
    d_acc, d_pre, d_rec, d_f1 = [], [], [], []
    for _ in tqdm(range(args.nb_draw)):
        draw = np.random.randint(0, 2, total)

        d_acc.append(accuracy_score(labels, draw))
        d_pre.append(precision_score(labels, draw, average="binary"))
        d_rec.append(recall_score(labels, draw, average="binary"))
        d_f1.append(f1_score(labels, draw, average="binary"))

    d_acc = np.mean(d_acc)
    d_pre = np.mean(d_pre)
    d_rec = np.mean(d_rec)
    d_f1 = np.mean(d_f1)

    e_acc = f"{e_acc * 100:5.2f}"
    e_pre = f"{e_pre * 100:5.2f}"
    e_rec = f"{e_rec * 100:5.2f}"
    e_f1 = f"{e_f1 * 100:5.2f}"
    d_acc = f"{d_acc * 100:5.2f}"
    d_pre = f"{d_pre * 100:5.2f}"
    d_rec = f"{d_rec * 100:5.2f}"
    d_f1 = f"{d_f1 * 100:5.2f}"

    msg = f"\n* CLASSIFICATION *\n"
    header = f"{'':^10} | Accuracy | Precision | Recall | F1 Score\n"
    msg += header
    msg += f"{'-' * len(header)}\n"
    msg += f"{'Expected':^10} | {e_acc:^8} | {e_pre:^9} | {e_rec:^6} | {e_f1:^8}\n"
    msg += f"{'-' * len(header)}\n"
    msg += f"{' Estimated':^10} | {d_acc:^8} | {d_pre:^9} | {d_rec:^6} | {d_f1:^8}\n"
    print(msg)
    ####################################################################################################################

    # Ranking
    ####################################################################################################################
    kmetric = [3]
    k_acc = kmetric
    k_pre_and_rec = kmetric
    k_ndcg = kmetric
    k_mrr = kmetric
    k_map = kmetric

    e_acc_k, d_acc_k = {k: [] for k in k_acc}, {k: [] for k in k_acc}
    e_pre_k, d_pre_k = {k: [] for k in k_pre_and_rec}, {k: [] for k in k_pre_and_rec}
    e_rec_k, d_rec_k = {k: [] for k in k_pre_and_rec}, {k: [] for k in k_pre_and_rec}
    e_ndcg_k, d_ndcg_k = {k: [] for k in k_ndcg}, {k: [] for k in k_ndcg}
    e_mrr_k, d_mrr_k = {k: [] for k in k_mrr}, {k: [] for k in k_mrr}
    e_map_k, d_map_k = {k: [] for k in k_map}, {k: [] for k in k_map}

    for app in tqdm(split.values()):
        sorted_cand = app.sort_values(by="label", ascending=False)
        relevant = sorted_cand[sorted_cand["label"] > 0]
        irrelevant = sorted_cand[sorted_cand["label"] == 0]
        nb_relevant = len(relevant)
        nb_irrelevant = len(irrelevant)
        tot_cand = nb_relevant + nb_irrelevant

        if not nb_relevant or not nb_irrelevant:
            continue

        ids = app["Id_candidate"]
        rel = app["label"]

        relevant_ids = relevant["Id_candidate"].values
        relevant_rel = relevant["label"].tolist() # original labels (0, 1, 2)

        # E[acc@k(j)]
        for k in e_acc_k:
            K = min(k, tot_cand)
            e_acc_k[k].append(expected_random_acc_at_k(tot_cand, nb_relevant, K))

        # E[pre@k(j)]
        for k in e_pre_k:
            K = min(k, tot_cand)
            e_pre_k[k].append((K * nb_relevant) / (k * tot_cand))

        # E[rec@k(j)]
        for k in e_rec_k:
            K = min(k, tot_cand)
            e_rec_k[k].append(K / tot_cand)

        # E[NDCG@k(j)]
        for k in e_ndcg_k:
            K = min(k, tot_cand)
            Q = min(k, nb_relevant)
            IDCG = 0
            E_discount = 0
            for i in range(K):
                discount = 1 / np.log2(i + 2)
                if i < Q:
                    IDCG += relevant_rel[i] * discount
                E_discount += discount
            E_gain = rel.sum()
            E_DCG = 1 / tot_cand * E_gain * E_discount
            e_ndcg_k[k].append(E_DCG / IDCG)

        # E[MRR@k(j)]
        for k in e_mrr_k:
            K = min(k, tot_cand)
            mrr = 0
            for i in range(1, K + 1):
                mrr += (1 / i) * math.comb(tot_cand - i, nb_relevant - 1)
            e_mrr_k[k].append(mrr / math.comb(tot_cand, nb_relevant))

        # E[MAP@k(j)]
        for k in e_map_k:
            K = min(k, tot_cand)
            Q = min(k, nb_relevant)
            r = (nb_relevant - 1) / (tot_cand - 1)
            map = 0
            for i in range(1, K + 1):
                map += (1 + (i - 1) * r) / i
            e_map_k[k].append(nb_relevant / (Q * tot_cand) * map)

        # Mean variables for the draws
        num_hits_at_k = {k: [] for k in k_acc}
        precisions_at_k = {k: [] for k in k_pre_and_rec}
        recall_at_k = {k: [] for k in k_pre_and_rec}
        MRR = {k: [] for k in k_mrr}
        ndcg = {k: [] for k in k_ndcg}
        MAP = {k: [] for k in k_map}
        for _ in range(args.nb_draw):
            random_perm = np.random.permutation(len(ids))
            random_preds = ids.values[random_perm]
            random_relevances = rel.values[random_perm]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k in k_acc:
                temp = 0
                for hit in random_preds[0:k]:
                    if hit in relevant_ids:
                        temp = 1
                        break
                num_hits_at_k[k].append(temp)

            # Precision and Recall@k
            for k in k_pre_and_rec:
                num_correct = 0
                for hit in random_preds[0:k]:
                    if hit in relevant_ids:
                        num_correct += 1

                precisions_at_k[k].append(num_correct / k)
                recall_at_k[k].append(num_correct / nb_relevant)

            # MRR@k
            for k in k_mrr:
                temp = 0
                for rank, hit in enumerate(random_preds[0:k]):
                    if hit in relevant_ids:
                        temp = 1.0 / (rank + 1)
                        break
                MRR[k].append(temp)

            # NDCG@k
            for k in k_ndcg:
                ndcg_value = compute_dcg_at_k(random_relevances, k) / compute_dcg_at_k(relevant_rel, k)
                ndcg[k].append(ndcg_value)

            # MAP@k
            for k in k_map:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(random_preds[0:k]):
                    if hit in relevant_ids:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k, nb_relevant)
                MAP[k].append(avg_precision)

        # Compute draw's average for 1 job
        for k in num_hits_at_k:
            d_acc_k[k].append(np.mean(num_hits_at_k[k]).item())

        for k in precisions_at_k:
            d_pre_k[k].append(np.mean(precisions_at_k[k]).item())

        for k in recall_at_k:
            d_rec_k[k].append(np.mean(recall_at_k[k]).item())

        for k in ndcg:
            d_ndcg_k[k].append(np.mean(ndcg[k]).item())

        for k in MRR:
            d_mrr_k[k].append(np.mean(MRR[k]).item())

        for k in MAP:
            d_map_k[k].append(np.mean(MAP[k]).item())

    # Compute averages for all the jobs
    # Expected
    for k in e_acc_k:
        e_acc_k[k] = np.mean(e_acc_k[k]).item()

    for k in e_pre_k:
        e_pre_k[k] = np.mean(e_pre_k[k]).item()

    for k in e_rec_k:
        e_rec_k[k] = np.mean(e_rec_k[k]).item()

    for k in e_ndcg_k:
        e_ndcg_k[k] = np.mean(e_ndcg_k[k]).item()

    for k in e_mrr_k:
        e_mrr_k[k] = np.mean(e_mrr_k[k]).item()

    for k in e_map_k:
        e_map_k[k] = np.mean(e_map_k[k]).item()

    # Estimated
    for k in d_acc_k:
        d_acc_k[k] = np.mean(d_acc_k[k]).item()

    for k in d_pre_k:
        d_pre_k[k] = np.mean(d_pre_k[k]).item()

    for k in d_rec_k:
        d_rec_k[k] = np.mean(d_rec_k[k]).item()

    for k in d_ndcg_k:
        d_ndcg_k[k] = np.mean(d_ndcg_k[k]).item()

    for k in d_mrr_k:
        d_mrr_k[k] = np.mean(d_mrr_k[k]).item()

    for k in d_map_k:
        d_map_k[k] = np.mean(d_map_k[k]).item()

    msg = "\n*RANKING*\n"
    msg_k_acc = f""
    msg_k_pre_and_rec = f""
    msg_k_ndcg = f""
    msg_k_mrr = f""
    msg_k_map = f""
    for k in k_acc:
        msg_k_acc += f" {str(k):>6} "
    for k in k_pre_and_rec:
        msg_k_pre_and_rec += f" {str(k):>6} "
    for k in k_ndcg:
        msg_k_ndcg += f" {str(k):>6} "
    for k in k_mrr:
        msg_k_mrr += f" {str(k):>6} "
    for k in k_map:
        msg_k_map += f" {str(k):>6} "

    header1 = (f"{'':^10} | {'HR@k':^{len(msg_k_acc) - 2}} | {'Pre@k':^{len(msg_k_pre_and_rec) - 2}} | "
               f"{'Rec@k':^{len(msg_k_pre_and_rec) - 2}} | {'NDCG@k':^{len(msg_k_ndcg) - 2}} | "
               f"{'MRR@k':^{len(msg_k_mrr) - 2}} | {'MAP@k':^{len(msg_k_map) - 2}}\n")
    msg += header1
    header2 = f"{'':^10} |{msg_k_acc}|{msg_k_pre_and_rec}|{msg_k_pre_and_rec}|{msg_k_ndcg}|{msg_k_mrr}|{msg_k_map}\n"
    msg += header2
    msg += f"{'-' * len(header1)}\n"
    msg += f"{'Expected':^10} |"
    for val in e_acc_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in e_pre_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in e_rec_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in e_ndcg_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in e_mrr_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in e_map_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"\n{'-' * len(header1)}\n"
    msg += f"{' Estimated':^10} |"
    for val in d_acc_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in d_pre_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in d_rec_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in d_ndcg_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in d_mrr_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "
    msg += f"|"
    for val in d_map_k.values():
        score = f"{val * 100:5.2f}"
        msg += f" {score:>6} "

    print(msg)
