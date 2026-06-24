"""
ml_predictor.py — Machine-learning prediction pipeline.

This module loads the pre-trained scikit-learn models (``.pkl`` files) and
runs the two-stage hierarchical classification:

    **Stage 1 — Class prediction**
        Four binary Balanced Random Forest (BRF) models (one per class) are
        evaluated.  The class with the highest positive-class probability is
        selected.

    **Stage 2 — Serogroup prediction**
        The serogroup-level OvR (One-vs-Rest) BRF models that belong to the
        predicted class are evaluated.  The serogroup with the highest
        probability wins.

The final output is a list of dictionaries (one per genome) ready to be
converted into a summary CSV.

Functions
---------
predict_class
    Run the class-level prediction for a single genome.
predict_serogroup
    Run the serogroup-level prediction given a predicted class.
run_ml_pipeline
    Orchestrate both stages for a batch of genomes.
"""

from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from .feature_extraction import format_features_for_model, get_features_from_list

# Suppress sklearn version-mismatch warnings that occur when loading
# pickled models trained with a slightly different scikit-learn release.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ── Class prediction ────────────────────────────────────────────────────────

def predict_class(
    extracted_scores: Dict[str, float],
    model_dir: str,
    seroclass_tsv: str,
    num_classes: int = 4,
) -> Tuple[Optional[str], Dict[str, float]]:
    """Predict the phylogenetic class for a single genome.

    Iterates over class indices ``1..num_classes``, loads the corresponding
    BRF model, and returns the class with the highest positive-class
    probability.

    Parameters
    ----------
    extracted_scores : dict[str, float]
        BLAST-derived protein scores (from :func:`blast_runner.run_tblastn`).
    model_dir : str
        Directory containing the ``.pkl`` model files.
    seroclass_tsv : str
        Path to the TSV that maps class names → expected feature lists.
    num_classes : int, optional
        Number of classes to evaluate (default ``4``).

    Returns
    -------
    tuple[str | None, dict[str, float]]
        A tuple of ``(predicted_class_number_str, class_probabilities)``.
        Returns ``(None, {})`` if no model could be evaluated.
    """
    class_probs: Dict[str, float] = {}

    for idx in range(1, num_classes + 1):
        # Support both naming conventions found in the repository
        candidate_names = [
            f"class_{idx}_brf_model.pkl",
            f"brf_model_class_{idx}.pkl",
        ]
        model_path = next(
            (
                os.path.join(model_dir, name)
                for name in candidate_names
                if os.path.exists(os.path.join(model_dir, name))
            ),
            None,
        )

        if model_path is None:
            continue

        column_name = f"Class {idx}"
        expected_features = get_features_from_list(seroclass_tsv, column_name)
        if not expected_features:
            continue

        try:
            model = joblib.load(model_path)
            x_input = format_features_for_model(extracted_scores, expected_features)

            # Re-align columns if the model stores its own feature names
            if hasattr(model, "feature_names_in_"):
                x_input = x_input.reindex(
                    columns=model.feature_names_in_, fill_value=0.0
                )

            prob = model.predict_proba(x_input)[0][1]
            class_probs[column_name] = prob
            print(f"    Class {idx} probability: {prob:.4f}")

        except Exception as exc:
            print(f"    [ERROR] Class {idx}: {exc}")
            continue

    if not class_probs:
        return None, {}

    predicted = max(class_probs, key=class_probs.get)  # e.g. "Class 2"
    predicted_num = predicted.split(" ")[1]
    return predicted_num, class_probs


# ── Serogroup prediction ───────────────────────────────────────────────────

def predict_serogroup(
    extracted_scores: Dict[str, float],
    model_dir: str,
    serogroups_tsv: str,
    predicted_class_num: str,
) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Predict the serogroup within the given class.

    Searches for OvR BRF models matching the pattern
    ``brf_ovr_class<N>_<Serogroup>.pkl`` and evaluates each one.

    Parameters
    ----------
    extracted_scores : dict[str, float]
        BLAST-derived protein scores.
    model_dir : str
        Directory containing the ``.pkl`` model files.
    serogroups_tsv : str
        Path to the TSV mapping serogroup names → expected feature lists.
    predicted_class_num : str
        The class number predicted in stage 1 (e.g. ``"2"``).

    Returns
    -------
    tuple[str | None, float, dict[str, float]]
        ``(serogroup_name, confidence, all_serogroup_probabilities)``.
        Returns ``(None, 0.0, {})`` if no model could be evaluated.
    """
    sg_probs: Dict[str, float] = {}

    pattern = os.path.join(model_dir, f"brf_ovr_class{predicted_class_num}_*.pkl")
    sg_model_paths = glob.glob(pattern)

    if not sg_model_paths:
        print(
            f"  [WARN] No serogroup models found for class {predicted_class_num} "
            f"(pattern: {pattern})"
        )
        return None, 0.0, {}

    print(
        f"  [ML] Evaluating {len(sg_model_paths)} serogroup model(s) "
        f"for class {predicted_class_num} ..."
    )

    for sg_path in sg_model_paths:
        sg_name = os.path.basename(sg_path).split("_")[-1].replace(".pkl", "")
        expected_features = get_features_from_list(serogroups_tsv, sg_name)

        if not expected_features:
            continue

        try:
            model = joblib.load(sg_path)
            x_input = format_features_for_model(extracted_scores, expected_features)

            if hasattr(model, "feature_names_in_"):
                x_input = x_input.reindex(
                    columns=model.feature_names_in_, fill_value=0.0
                )

            prob = model.predict_proba(x_input)[0][1]
            sg_probs[sg_name] = prob
            print(f"    Serogroup {sg_name}: {prob:.4f}")

        except Exception as exc:
            print(f"    [ERROR] Serogroup {sg_name}: {exc}")
            continue

    if not sg_probs:
        return None, 0.0, {}

    best_sg = max(sg_probs, key=sg_probs.get)
    return best_sg, sg_probs[best_sg], sg_probs


# ── Full pipeline (batch) ──────────────────────────────────────────────────

def run_ml_pipeline(
    all_scores: Dict[str, Dict[str, float]],
    model_dir: str,
    data_dir: str,
) -> List[Dict[str, Any]]:
    """Run the complete two-stage ML pipeline for every genome in the batch.

    Parameters
    ----------
    all_scores : dict[str, dict[str, float]]
        Mapping of ``{genome_path: {protein: score}}`` as produced by the
        BLAST stage.
    model_dir : str
        Path to the directory containing all ``.pkl`` model files.
    data_dir : str
        Path to the directory containing the training TSV files
        (``seroclasses_to_train.tsv`` and ``serogroups_to_train.tsv``).

    Returns
    -------
    list[dict]
        A list of result dictionaries, one per genome, with keys:
        ``Genome``, ``Predicted_Class``, ``Predicted_Serogroup``,
        ``Confidence``.
    """
    seroclass_tsv = os.path.join(data_dir, "seroclasses_to_train.tsv")
    serogroups_tsv = os.path.join(data_dir, "serogroups_to_train.tsv")

    results: List[Dict[str, Any]] = []

    for genome_path, scores in all_scores.items():
        genome_name = os.path.basename(genome_path)
        print(f"\n{'─' * 50}")
        print(f"  ML pipeline — {genome_name}")
        print(f"  Proteins with BLAST hits: {len(scores)}")
        print(f"{'─' * 50}")

        # Stage 1: class prediction
        class_num, _ = predict_class(scores, model_dir, seroclass_tsv)
        if class_num is None:
            print(f"  [FAIL] No class could be predicted for {genome_name}.")
            continue

        print(f"  → Predicted class: {class_num}")

        # Stage 2: serogroup prediction
        sg_name, confidence, _ = predict_serogroup(
            scores, model_dir, serogroups_tsv, class_num
        )
        if sg_name is None:
            print(f"  [FAIL] No serogroup predicted for class {class_num}.")
            continue

        print(f"  → Predicted serogroup: {sg_name} ({confidence * 100:.1f}%)")

        results.append(
            {
                "Genome": genome_name,
                "Predicted_Class": class_num,
                "Predicted_Serogroup": sg_name,
                "Confidence": f"{confidence * 100:.1f}%",
            }
        )

    return results
