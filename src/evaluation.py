"""
Evaluation module implementing exact competition entity-level Macro F0.5 metric.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np

def calculate_single_entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F0.5 for a single Source 1 entity.
    Follows exact competition singleton and macro-averaging specifications:
    - If GT is empty and Pred is empty: F0.5 = 1.0, Precision = 1.0, Recall = 1.0
    - If GT is empty and Pred is not empty: F0.5 = 0.0, Precision = 0.0, Recall = 1.0
    - If GT is not empty and Pred is empty: F0.5 = 0.0, Precision = 1.0, Recall = 0.0
    - Else: standard Precision, Recall, and F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if not gt_ids and not pred_ids:
        return 1.0, 1.0, 1.0
    if not gt_ids and pred_ids:
        return 0.0, 0.0, 1.0
    if gt_ids and not pred_ids:
        return 0.0, 1.0, 0.0

    tp = len(gt_ids & pred_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (0.25 * precision + recall)
    if denom > 0:
        f05 = (1.25 * precision * recall) / denom
    else:
        f05 = 0.0

    return f05, precision, recall

def evaluate_predictions(
    gt_mapping: Dict[str, List[str]],
    pred_mapping: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Evaluates predictions across all S1 entities in gt_mapping and computes Macro F0.5.
    """
    f05_list = []
    prec_list = []
    rec_list = []
    pred_match_counts = []
    correct_singletons = 0
    total_singletons = 0

    for s1_id, gt_list in gt_mapping.items():
        gt_set = set(gt_list)
        pred_set = set(pred_mapping.get(s1_id, []))
        pred_match_counts.append(len(pred_set))

        f05, prec, rec = calculate_single_entity_f05(gt_set, pred_set)
        f05_list.append(f05)
        prec_list.append(prec)
        rec_list.append(rec)

        if not gt_set:
            total_singletons += 1
            if not pred_set:
                correct_singletons += 1

    macro_f05 = float(np.mean(f05_list)) if f05_list else 0.0
    macro_prec = float(np.mean(prec_list)) if prec_list else 0.0
    macro_rec = float(np.mean(rec_list)) if rec_list else 0.0
    avg_pred_matches = float(np.mean(pred_match_counts)) if pred_match_counts else 0.0
    singleton_acc = (correct_singletons / total_singletons) if total_singletons > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "avg_predicted_matches": avg_pred_matches,
        "singleton_accuracy": singleton_acc,
        "total_singletons": total_singletons,
        "correct_singletons": correct_singletons,
        "evaluated_s1_count": len(gt_mapping)
    }

def find_best_threshold(
    gt_mapping: Dict[str, List[str]],
    val_cand_scores: Dict[str, List[Tuple[str, float]]],
    threshold_grid: List[float]
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    """
    Searches threshold grid to find optimal decision threshold on validation set.
    """
    best_thresh = 0.50
    best_metrics = None
    all_results = []

    print("-" * 75)
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Avg Matches':<12} | {'Singl Acc':<10}")
    print("-" * 75)

    for thresh in threshold_grid:
        pred_map = {}
        for s1_id in gt_mapping.keys():
            scores = val_cand_scores.get(s1_id, [])
            passed = [cand_id for cand_id, prob in scores if prob >= thresh]
            pred_map[s1_id] = passed

        metrics = evaluate_predictions(gt_mapping, pred_map)
        metrics["threshold"] = thresh
        all_results.append(metrics)

        print(f"{thresh:<10.2f} | {metrics['macro_f05']:<12.4f} | {metrics['macro_precision']:<10.4f} | {metrics['macro_recall']:<10.4f} | {metrics['avg_predicted_matches']:<12.2f} | {metrics['singleton_accuracy']*100:<9.1f}%")

        if best_metrics is None or metrics["macro_f05"] > best_metrics["macro_f05"]:
            best_metrics = metrics
            best_thresh = thresh

    print("-" * 75)
    print(f"Optimal Threshold Selected: {best_thresh:.2f} (Macro F0.5 = {best_metrics['macro_f05']:.4f})")
    print("-" * 75)

    return best_thresh, best_metrics, all_results
