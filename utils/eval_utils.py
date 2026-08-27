
"""
eval_utils.py
Top-K accuracy, precision@k, recall@k, and mAP for zero-shot mask classification.
Assumes you have per-mask logits and ground-truth class ids.
"""

from typing import List, Dict, Tuple, Iterable
import numpy as np

def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    idx = np.argsort(-logits, axis=1)[:, :k]
    hits = sum(t in idx[i] for i, t in enumerate(targets))
    return float(hits) / len(targets) if len(targets) else 0.0

def precision_at_k(logits: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    idx = np.argsort(-logits, axis=1)[:, :k]
    correct = [(t in idx[i]) for i, t in enumerate(targets)]
    return float(sum(correct)) / (k * len(targets)) * k if len(targets) else 0.0

def average_precision_for_query(binary_relevance: Iterable[int]) -> float:
    hits, s = 0, 0.0
    for i, rel in enumerate(binary_relevance, start=1):
        if rel:
            hits += 1
            s += hits / i
    return s / max(hits, 1)

def mean_average_precision(logits: np.ndarray, targets: np.ndarray) -> float:
    # convert each row into ranked binary relevance
    mAPs = []
    for i in range(logits.shape[0]):
        order = np.argsort(-logits[i])
        binary = [1 if c == targets[i] else 0 for c in order]
        mAPs.append(average_precision_for_query(binary))
    return float(np.mean(mAPs)) if mAPs else 0.0
