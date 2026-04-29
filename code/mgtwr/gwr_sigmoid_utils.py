from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


EPS = 1.0e-12
DEFAULT_SIGMOID_CLIP = 35.0
DEFAULT_ROBUST_CLIP_Z = 6.0


def sigmoid(values: np.ndarray, clip: float = DEFAULT_SIGMOID_CLIP) -> np.ndarray:
    """Standard sigmoid."""
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -float(clip), float(clip))
    return 1.0 / (1.0 + np.exp(-values))


def fit_robust_sigmoid_iqr_transform(
    scores: np.ndarray,
    clip_z: float = DEFAULT_ROBUST_CLIP_Z,
) -> Dict[str, float]:
    """Fit robust sigmoid parameters with median + IQR."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise ValueError("There are no finite GWR output values ​​available for fitting robust sigmoid.")

    center = float(np.nanmedian(finite_scores))
    q25, q75 = np.nanpercentile(finite_scores, [25, 75])
    iqr = float(q75 - q25)

    scale = iqr / (2.0 * np.log(3.0)) if np.isfinite(iqr) and iqr > EPS else np.nan
    if (not np.isfinite(scale)) or scale <= EPS:
        scale = float(np.nanstd(finite_scores))
    if (not np.isfinite(scale)) or scale <= EPS:
        scale = 1.0

    return {
        "method": "robust_sigmoid_iqr",
        "center": center,
        "scale": float(scale),
        "clip_z": float(clip_z),
    }


def robust_sigmoid_transform(
    scores: np.ndarray,
    center: float,
    scale: float,
    clip_z: float = DEFAULT_ROBUST_CLIP_Z,
) -> np.ndarray:
    """Map raw fraction to 0-1 at given center/scale/clip_z."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    probabilities = np.full(scores.shape, np.nan, dtype=float)
    valid = np.isfinite(scores)
    if not valid.any():
        raise ValueError("No finite value available in scores.")

    scale = float(scale)
    if (not np.isfinite(scale)) or scale <= EPS:
        scale = 1.0

    z_scores = (scores[valid] - float(center)) / scale
    z_scores = np.clip(z_scores, -float(clip_z), float(clip_z))
    probabilities[valid] = sigmoid(z_scores)
    return probabilities


def convert_gwr_scores_to_susceptibility_probabilities(
    gwr_results: np.ndarray,
    transform_metadata: Optional[Dict[str, float]] = None,
    clip_z: float = DEFAULT_ROBUST_CLIP_Z,
    return_metadata: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]] | np.ndarray:
    """Map GWR raw regression output to 0-1 probabilities."""
    scores = np.asarray(gwr_results, dtype=float).reshape(-1)
    if transform_metadata is None:
        transform_metadata = fit_robust_sigmoid_iqr_transform(scores, clip_z=clip_z)

    method = transform_metadata.get("method")
    if method != "robust_sigmoid_iqr":
        raise ValueError(f"Unsupported transform method:{method}")

    metadata = {
        "method": "robust_sigmoid_iqr",
        "center": float(transform_metadata["center"]),
        "scale": float(transform_metadata["scale"]),
        "clip_z": float(transform_metadata.get("clip_z", clip_z)),
    }
    probabilities = robust_sigmoid_transform(
        scores,
        center=metadata["center"],
        scale=metadata["scale"],
        clip_z=metadata["clip_z"],
    )

    if return_metadata:
        return probabilities, metadata
    return probabilities


def gwr_scores_to_probabilities(
    gwr_results: np.ndarray,
    transform_metadata: Optional[Dict[str, float]] = None,
    clip_z: float = DEFAULT_ROBUST_CLIP_Z,
    return_metadata: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]] | np.ndarray:
    """`convert_gwr_scores_to_susceptibility_probabilities` ."""
    return convert_gwr_scores_to_susceptibility_probabilities(
        gwr_results,
        transform_metadata=transform_metadata,
        clip_z=clip_z,
        return_metadata=return_metadata,
    )


def to_binary_labels(labels: np.ndarray, positive_cutoff: float = 0.55) -> np.ndarray:
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if not np.isfinite(labels).all():
        raise ValueError("The label contains non-finite values and cannot be evaluated for binary classification.")
    return (labels > float(positive_cutoff)).astype(int)


def summarize_values(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
    }


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    valid_mask = np.isfinite(probabilities)
    if valid_mask.sum() == 0:
        return float("nan")
    y_true = y_true[valid_mask]
    probabilities = probabilities[valid_mask]
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def select_probability_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    valid_mask = np.isfinite(probabilities)
    if valid_mask.sum() == 0 or np.unique(y_true[valid_mask]).size < 2:
        return 0.5, {"youden": float("nan"), "fpr": float("nan"), "tpr": float("nan")}

    fpr, tpr, thresholds = roc_curve(y_true[valid_mask], probabilities[valid_mask])
    youden = tpr - fpr
    finite_mask = np.isfinite(thresholds)
    if finite_mask.any():
        best_idx = int(np.argmax(np.where(finite_mask, youden, -np.inf)))
        threshold = float(thresholds[best_idx])
    else:
        best_idx = 0
        threshold = 0.5

    return threshold, {
        "youden": float(youden[best_idx]),
        "fpr": float(fpr[best_idx]),
        "tpr": float(tpr[best_idx]),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan}

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def fit_gwr_probability_classifier(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    label_cutoff: float = 0.55,
    validation_size: float = 0.2,
    random_state: int = 42,
    clip_z: float = DEFAULT_ROBUST_CLIP_Z,
) -> Dict[str, Any]:
    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    probabilities, transform_metadata = gwr_scores_to_probabilities(raw_scores, clip_z=clip_z)
    y_binary = to_binary_labels(labels, positive_cutoff=label_cutoff)
    valid_mask = np.isfinite(probabilities)

    if valid_mask.sum() < 10:
        raise ValueError("There are too few effective GWR training probabilities for threshold selection.")

    y_eval = y_binary[valid_mask]
    prob_eval = probabilities[valid_mask]
    _, validation_probabilities, _, y_valid = train_test_split(
        prob_eval,
        y_eval,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_eval,
    )

    probability_threshold, threshold_info = select_probability_threshold(y_valid, validation_probabilities)
    predicted_classes = np.full(probabilities.shape, -1, dtype=int)
    predicted_classes[valid_mask] = (probabilities[valid_mask] >= probability_threshold).astype(int)
    validation_predictions = (validation_probabilities >= probability_threshold).astype(int)

    train_binary_metrics = {
        "accuracy": float(accuracy_score(y_eval, predicted_classes[valid_mask])),
        "roc_auc": safe_roc_auc(y_eval, prob_eval),
    }
    validation_binary_metrics = {
        "accuracy": float(accuracy_score(y_valid, validation_predictions)),
        "roc_auc": safe_roc_auc(y_valid, validation_probabilities),
    }
    train_regression = regression_metrics(labels, raw_scores)

    artifacts = {
        "label_cutoff": float(label_cutoff),
        "validation_size": float(validation_size),
        "random_state": int(random_state),
        "probability_source": "raw_gwr_score_probability_transform",
        "transform_metadata": transform_metadata,
        "probability_threshold": float(probability_threshold),
        "threshold_selection": "validation_youden_j",
        "threshold_info": threshold_info,
        "train_raw_score_stats": summarize_values(raw_scores),
        "train_probability_stats": summarize_values(probabilities),
        "train_regression_metrics": train_regression,
        "train_binary_metrics": train_binary_metrics,
        "validation_binary_metrics": validation_binary_metrics,
    }

    return {
        "raw_scores": raw_scores,
        "probabilities": probabilities,
        "predicted_classes": predicted_classes,
        "valid_mask": valid_mask,
        "probability_threshold": float(probability_threshold),
        "transform_metadata": transform_metadata,
        "threshold_info": threshold_info,
        "train_regression_metrics": train_regression,
        "train_binary_metrics": train_binary_metrics,
        "validation_binary_metrics": validation_binary_metrics,
        "artifacts": artifacts,
    }


def predict_gwr_probability_classifier(
    raw_scores: np.ndarray,
    artifacts: Dict[str, Any],
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    transform_metadata = artifacts.get("transform_metadata")
    probability_threshold = float(artifacts.get("probability_threshold", 0.5))
    label_cutoff = float(artifacts.get("label_cutoff", 0.55))

    probabilities, transform_metadata = gwr_scores_to_probabilities(raw_scores, transform_metadata)
    valid_mask = np.isfinite(probabilities)
    predicted_classes = np.full(probabilities.shape, -1, dtype=int)
    predicted_classes[valid_mask] = (probabilities[valid_mask] >= probability_threshold).astype(int)

    result: Dict[str, Any] = {
        "raw_scores": raw_scores,
        "probabilities": probabilities,
        "predicted_classes": predicted_classes,
        "valid_mask": valid_mask,
        "probability_threshold": probability_threshold,
        "transform_metadata": transform_metadata,
        "label_cutoff": label_cutoff,
        "evaluation": None,
    }

    if y_true is None:
        return result

    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    eval_mask = np.isfinite(probabilities) & np.isfinite(y_true_array)
    if eval_mask.sum() == 0:
        result["evaluation"] = {
            "available": False,
            "skip_reason": "no_finite_values",
        }
        return result

    raw_eval = raw_scores[eval_mask]
    prob_eval = probabilities[eval_mask]
    pred_eval = predicted_classes[eval_mask]
    y_eval = y_true_array[eval_mask]
    y_binary_eval = to_binary_labels(y_eval, positive_cutoff=label_cutoff)

    evaluation: Dict[str, Any] = {
        "available": True,
        "skip_reason": None,
        "eval_mask": eval_mask,
        "regression_metrics": regression_metrics(y_eval, raw_eval),
        "binary_metrics": None,
        "confusion_matrix": None,
        "classification_report": None,
    }

    if np.unique(y_binary_eval).size < 2:
        evaluation["skip_reason"] = "single_class"
        result["evaluation"] = evaluation
        return result

    evaluation["binary_metrics"] = {
        "accuracy": float(accuracy_score(y_binary_eval, pred_eval)),
        "roc_auc": float(roc_auc_score(y_binary_eval, prob_eval)),
    }
    evaluation["confusion_matrix"] = confusion_matrix(y_binary_eval, pred_eval)
    evaluation["classification_report"] = classification_report(y_binary_eval, pred_eval)
    result["evaluation"] = evaluation
    return result


__all__ = [
    "sigmoid",
    "fit_robust_sigmoid_iqr_transform",
    "robust_sigmoid_transform",
    "convert_gwr_scores_to_susceptibility_probabilities",
    "gwr_scores_to_probabilities",
    "to_binary_labels",
    "summarize_values",
    "safe_roc_auc",
    "select_probability_threshold",
    "regression_metrics",
    "fit_gwr_probability_classifier",
    "predict_gwr_probability_classifier",
]
