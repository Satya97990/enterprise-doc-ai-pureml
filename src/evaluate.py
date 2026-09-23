# src/evaluate.py
import numpy as np
import evaluate
from transformers import EvalPrediction
from typing import Dict, Callable

# Load the seqeval metric for strict entity-level evaluation
metric = evaluate.load("seqeval")

def get_compute_metrics_fn(id2label: Dict[int, str]) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Returns a compute_metrics function configured with the provided label mapping.
    Filters out ignored tokens (-100) before calculating macro F1, Precision, and Recall.
    """
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens and sub-words flagged with -100)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics