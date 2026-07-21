import numpy as np
from collections import defaultdict


# Classification
def equal_opportunity(y_pred, y_gt, sensitive_attribute):
    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is positive.
    y_pred = y_pred[y_gt == 1]
    sensitive_attribute = sensitive_attribute[y_gt == 1]

    y_z_1 = y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0]

    # If there are no data points in one of the sensitive attribute groups, return 0.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    y_z_1 = y_z_1.mean()
    y_z_0 = y_z_0.mean()

    return (min(y_z_1, y_z_0) + 1e-8) / (max(y_z_1, y_z_0) + 1e-8)


def equalized_odds(y_pred, y_gt, sensitive_attribute):
    y_pred_all = y_pred.copy()
    sensitive_attribute_all = sensitive_attribute.copy()

    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is positive.
    y_pred = y_pred_all[y_gt == 1]
    sensitive_attribute = sensitive_attribute_all[y_gt == 1]

    # Calculate the difference in true positive rate.
    y_z_1 = y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    y_z_1 = y_z_1.mean()
    y_z_0 = y_z_0.mean()
    equality = (min(y_z_1, y_z_0) + 1e-8) / (max(y_z_1, y_z_0) + 1e-8)

    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is negative.
    y_pred = y_pred_all[y_gt == 0]
    sensitive_attribute = sensitive_attribute_all[y_gt == 0]

    # Calculate the difference in false positive rate.
    y_z_1 = y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    y_z_1 = y_z_1.mean()
    y_z_0 = y_z_0.mean()
    equality += (min(y_z_1, y_z_0) + 1e-8) / (max(y_z_1, y_z_0) + 1e-8)

    return equality / 2  # normalized between 0 and 1


def demographic_parity(y_pred, sensitive_attribute):
    y_z_1 = y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0]

    # If there are no data points in one of the sensitive attribute groups, return 0.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    y_z_1 = y_z_1.mean()
    y_z_0 = y_z_0.mean()

    return (min(y_z_1, y_z_0) + 1e-8) / (max(y_z_1, y_z_0) + 1e-8)


# Ranking
def group_exposure_and_utility(sa, relevance):
    exposure = np.array([1.0 / np.log2(2 + r) for r in range(len(sa))])
    group_exp = defaultdict(list)
    group_util = defaultdict(list)
    for i, group in enumerate(sa):
        group_exp[group].append(exposure[i])
        group_util[group].append(relevance[i])
    for group in group_exp.keys():
        group_exp[group] = np.mean(group_exp[group])
        group_util[group] = np.mean(group_util[group])
    return group_exp, group_util


def disparate_exposure_ratio(sa, relevance): # sa is already arranged in descending order
    exp, _ = group_exposure_and_utility(sa, relevance)
    values = np.array(list(exp.values()))
    return (values.min() + 1e-8) / (values.max() + 1e-8)


def disparate_treatment_ratio(sa, relevance):
    exp, util = group_exposure_and_utility(sa, relevance)
    exp = np.array(list(exp.values()))
    util = np.array(list(util.values()))
    dt = (exp + 1e-8) / (util + 1e-8)
    return dt.min() / dt.max()
