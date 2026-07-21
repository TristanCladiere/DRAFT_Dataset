import argparse
import torch
import numpy as np

from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from dataset import create_k_cross_dataset
from evaluator import RCFEvaluator, create_results_table
from summarize_k_cross_results import update_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='Data')
    parser.add_argument('--version', type=str, default='0.6B', choices=['0.6B', '4B', '8B'])
    _args = parser.parse_args()
    return _args


def leakage_evaluation(cand_emb: dict, sa: dict):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(cand_emb["Train"], sa["Train"])
    y_pred = clf.predict(cand_emb["Test"])
    return accuracy_score(sa["Test"], y_pred)*100


def get_best_threshold(scores: list, lab: torch.Tensor):
    best_threshold = -1
    rows = list(zip(scores, lab))
    rows = sorted(rows, key=lambda x: x[0], reverse=True)

    best_f1 = 0
    nextract = 0
    ncorrect = 0
    remaining_negatives = sum(lab == 0)
    total_pos = sum(lab)

    for i in range(len(rows) - 1):
        score, lab = rows[i]
        nextract += 1

        if lab == 1:
            ncorrect += 1
        else:
            remaining_negatives -= 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_pos
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2
    return best_threshold.item()


if __name__ == '__main__':
    args = parse_args()
    version_to_emb_size = {"0.6B": 1024, "4B": 2560, "8B": 4096}
    emb_size = version_to_emb_size[args.version]

    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    leakages = []
    for k_idx in tqdm(range(1,6)):
        test_data, pairs, labels, sas = create_k_cross_dataset(
        args.data_folder, args.version, k_idx - 1, 5, for_cos_sim=True
        )

        # Cosine similarity evaluation
        train_sims = [(p[:emb_size] * p[emb_size:emb_size*2]).sum() for p in pairs["Train"]] # cosine similarity
        threshold = get_best_threshold(train_sims, torch.stack(labels["Train"])) # threshold maximizing F1-score on train set

        test_sims = [(p[:emb_size] * p[emb_size:emb_size*2]).sum() for p in pairs["Test"]]
        test_preds = [1 if sim >= threshold else 0 for sim in test_sims]

        evaluator = RCFEvaluator(ranking_data=test_data,
                                 classif_data=labels["Test"],
                                 mrr_at_k=[3],
                                 ndcg_at_k=[3],
                                 hr_at_k=[3],
                                 precision_recall_at_k=[3],
                                 map_at_k=[3])

        results, txt_job_rank, txt_job_fair = evaluator(test_sims, test_preds)
        summary = update_summary(summary, results, k_idx-1)

        # Leakage
        train_cand_emb = torch.stack(pairs["Train"])[:, emb_size:emb_size * 2]
        test_cand_emb = pairs["Test"][:, emb_size:emb_size * 2]
        leakages.append(leakage_evaluation({"Train": train_cand_emb, "Test": test_cand_emb}, sas))

    print(create_results_table(summary, add_margin=True))
    print('')
    print(f"Average Leakage (classification % accuracy): {np.mean(leakages):5.2f} +/-{np.std(leakages):5.2f}")