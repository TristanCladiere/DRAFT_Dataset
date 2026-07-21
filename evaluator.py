import torch
import heapq
import json
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fairness_metrics import (demographic_parity, equal_opportunity, equalized_odds, disparate_exposure_ratio,
                              disparate_treatment_ratio)


class RCFEvaluator:  # Ranking Classification and Fairness
    def __init__(
            self,
            ranking_data: dict,  # {"job_id": DataFrame('Id_candidate', 'label', 'sa')}
            classif_data: list,  # test labels, already binarized (from {0, 1, 2} to {0, 1})
            mrr_at_k: list[int] = [5],
            ndcg_at_k: list[int] = [5],
            hr_at_k: list[int] = [1, 3, 5],
            precision_recall_at_k: list[int] = [1, 3, 5],
            map_at_k: list[int] = [5]
    ):
        self.ranking_data = ranking_data
        self.classif_data = classif_data

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.hr_at_k = hr_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

    def __call__(self, scores: list, preds: list, save_dir_path='', is_cos_sim=False):

        results = {"Classification": dict(), "Ranking": dict()}

        # Classification evaluation:
        labels = np.array(self.classif_data)
        classif = dict()
        avg = "binary"
        classif["Acc"] = accuracy_score(labels, preds)
        classif["F1"] = f1_score(labels, preds, average=avg)
        classif["Pre"] = precision_score(labels, preds, average=avg)
        classif["Rec"] = recall_score(labels, preds, average=avg)
        results["Classification"]["Utility"] = classif

        # Ranking evaluation:
        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.hr_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )
        jobs_result_list = [[] for _ in range(len(self.ranking_data))]

        scores_top_k_values, scores_top_k_idx = [], []
        fairness = dict()
        start = 0
        job_ids = []
        for job_id, cand_infos in self.ranking_data.items():
            job_ids.append(job_id)
            stop = start + len(cand_infos)
            job_scores = scores[start:stop]
            job_preds = preds[start:stop]

            # Save fairness related infos
            fairness[job_id] = list(
                zip(np.asarray(job_preds), np.asarray(job_scores), cand_infos.sa.values, cand_infos.label.values)
            )

            start = stop

            score_top_k_value, score_top_k_idx = torch.topk(
                torch.tensor(job_scores), min(max_k, len(job_scores)), dim=0, largest=True, sorted=False
            )
            scores_top_k_values.append(score_top_k_value)
            scores_top_k_idx.append(score_top_k_idx)

        for job_pos, job_id in enumerate(job_ids):
            for cand_pos, score in zip(scores_top_k_idx[job_pos], scores_top_k_values[job_pos]):
                cand_id = self.ranking_data[job_id]["Id_candidate"].iloc[cand_pos.item()]
                sa = self.ranking_data[job_id]["sa"].iloc[cand_pos.item()]
                if len(jobs_result_list[job_pos]) < max_k:
                    # heapq tracks the quantity of the first element in the tuple
                    heapq.heappush(jobs_result_list[job_pos], (score, cand_id, sa))
                else:
                    heapq.heappushpop(jobs_result_list[job_pos], (score, cand_id, sa))

        for job_itr in range(len(jobs_result_list)):
            for cand_itr in range(len(jobs_result_list[job_itr])):
                score, cand_id, sa = jobs_result_list[job_itr][cand_itr]
                jobs_result_list[job_itr][cand_itr] = {"cand_id": cand_id, "score": score, "sa": sa}

        # Init score computation values
        # Ranking metrics
        num_hits_at_k = {k: 0 for k in self.hr_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Fairness metrics
        DP = []
        EOpp = []
        EOdds = []
        DPR = []
        DT = []

        self.number_of_valuable_ranking_jobs = 0  # number of jobs with at least 1 relevant cand and 1 irrelevant cand, used for ranking evaluation
        for job_pos, job_id in enumerate(job_ids):
            job_sub_corpus = self.ranking_data[job_id].sort_values("label", ascending=False)  # pd.DataFrame('Id_candidate', 'label', 'sa'), sorted by relevance (most relevant has highest label)

            # We only keep as relevant the candidate(s) who is (are) labelled with label > 0.
            mask = job_sub_corpus["label"] > 0
            relevant = job_sub_corpus[mask]
            irrelevant = job_sub_corpus[~mask]

            # Ensure that we have both relevant and irrelevant candidates to have a fair evaluation: with only relevant
            # candidates, the model can't fail, while it will always fail with only irrelevant ones.
            # Just ignore those cases.
            if (not len(relevant)) or (not len(irrelevant)):
                continue
            else:
                self.number_of_valuable_ranking_jobs += 1

            # Sort predicted scores
            top_hits = sorted(jobs_result_list[job_pos], key=lambda x: x["score"], reverse=True)

            # Hit-Rate@k - We count the result correct, if at least one relevant cand is across the top-k predicted cands
            for k_val in self.hr_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["cand_id"] in relevant['Id_candidate'].values:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["cand_id"] in relevant['Id_candidate'].values:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(relevant))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["cand_id"] in relevant['Id_candidate'].values:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = []
                for top_hit in top_hits[0:k_val]:
                    ids_relevant = relevant['Id_candidate'].values
                    if top_hit["cand_id"] in ids_relevant:
                        # Instead of using relevance = 1, we use the original label (1 or 2)
                        relevance = int(relevant[relevant['Id_candidate']==top_hit["cand_id"]]['label'].values)
                    else:
                        relevance = 0
                    predicted_relevance.append(relevance)

                true_relevances = list(relevant['label'])

                ndcg_value = compute_dcg_at_k(predicted_relevance, k_val) / compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["cand_id"] in relevant['Id_candidate'].values:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(relevant))
                AveP_at_k[k_val].append(avg_precision)

        # Fairness
        self.nb_of_mixed_jobs = 0 # number of jobs with at least 1 male cand and 1 female cand, used for fairness evaluation
        for job_id, job_scores in fairness.items():  # job_scores = (preds, scores, sas, labels)
            sorted_scores = sorted(job_scores, key=lambda x: x[1], reverse=True)
            sa = np.asarray([t[2] for t in sorted_scores])
            relevances = np.asarray([t[3] for t in sorted_scores]) # original labels (0, 1 ,2)
            labels = (relevances > 0).astype(int)  # binary labels (0, 1)
            preds = np.asarray([t[0] for t in sorted_scores])

            if len(np.unique_values(sa)) < 2:  # only one gender for this job offer
                continue
            else:
                self.nb_of_mixed_jobs += 1

            DP.append(demographic_parity(preds, sa))
            EOpp.append(equal_opportunity(preds, labels, sa))
            EOdds.append(equalized_odds(preds, labels, sa))
            DPR.append(disparate_exposure_ratio(sa, relevances))
            DT.append(disparate_treatment_ratio(sa, relevances))

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= self.number_of_valuable_ranking_jobs

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k]).item()

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k]).item()

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k]).item()

        for k in MRR:
            MRR[k] /= self.number_of_valuable_ranking_jobs

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k]).item()

        DP = np.mean(DP).item()
        EOpp = np.mean(EOpp).item()
        EOdds = np.mean(EOdds).item()

        DPR = np.mean(DPR).item()
        DT = np.mean(DT).item()

        results["Ranking"]["Utility"] = {
            "HR@k": num_hits_at_k,
            "Pre@k": precisions_at_k,
            "Rec@k": recall_at_k,
            "NDCG@k": ndcg,
            "MRR@k": MRR,
            "MAP@k": AveP_at_k
        }
        results["Ranking"]["Fairness"] = {
            "DPR": DPR,
            "DT": DT,
        }
        results["Classification"]["Fairness"] = {
                "DP": DP,
                "EOpp": EOpp,
                "EOdds": EOdds
            }

        if save_dir_path:
            with open(f"{save_dir_path}/results", "w") as f:
                json.dump(results, f)

        return (results,
                f"-> Used jobs for ranking evaluation: {self.number_of_valuable_ranking_jobs}/{len(self.ranking_data)}",
                f"-> Used jobs for fairness evaluation: {self.nb_of_mixed_jobs}/{len(self.ranking_data)}")


def compute_dcg_at_k(relevances, k):
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg


def create_classif_table(classif_results, add_margin):
    table = ""
    header = "|"
    scores_txt = "|"
    w = 19 if add_margin else 5
    for k, v in classif_results.items():
        if add_margin:
            m = np.mean(v)
            margin = np.std(v)
            temp = fr"{m * 100:5.2f} (+/-{margin * 100:5.2f})"
        else:
            temp = f"{v * 100:5.2f}"
        header += f" {k:^{w}} |"
        scores_txt += f" {temp:^{w}} |"
    table += f"{'-' * len(header)}\n"
    table += f"{header}\n"
    table += f"{'-' * len(header)}\n"
    table += scores_txt
    table += f"\n{'-' * len(header)}\n"
    return table


def create_ranking_table(ranking_results, add_margin):
    table = ""
    header1 = "|"
    header2 = "|"
    scores_txt = "|"
    w = 19 if add_margin else 5
    for key, val in ranking_results.items():
        k_txt = ""
        sub_scores_txt = ""
        for k, v in val.items():
            if add_margin:
                m = np.mean(v)
                margin = np.std(v)
                temp = fr"{m * 100:5.2f} (+/-{margin * 100:5.2f})"
            else:
                temp = f"{v*100:5.2f}"
            sub_scores_txt += f" {temp:>{w}} "
            k_txt += f" {k:>{w}} "
        header2 += f"{k_txt}|"
        scores_txt += f"{sub_scores_txt}|"
        header1 += f"{key:^{len(k_txt)}}|"
    table += f"{'-' * len(header1)}\n"
    table += f"{header1}\n{header2}\n"
    table += f"{'-' * len(header1)}\n"
    table += scores_txt
    table += f"\n{'-' * len(header1)}\n"
    return table


def create_results_table(results, add_margin=False):
    table = f"Classification\n"
    for k, v in results["Classification"].items():
        table += f"  -{k}\n"
        table += f"{create_classif_table(v, add_margin)}\n"

    table += f"\nRanking\n  -Utility\n"
    table += create_ranking_table(results["Ranking"]["Utility"], add_margin)
    table += f"\n  -Fairness\n"
    table += create_classif_table(results["Ranking"]["Fairness"], add_margin)
    return table
