"""
feature_extraction.py — Transform raw BLAST scores into ML-ready feature matrices.

This module bridges the gap between the BLAST output (a dictionary of
``{protein_accession: score}``) and the scikit-learn models that expect a
fixed-width DataFrame with specific column names.

Two main operations are performed:

1.  **Reading expected feature lists** from the training TSV files shipped
    with the repository (e.g. ``seroclasses_to_train.tsv`` and
    ``serogroups_to_train.tsv``).  Each column in those TSVs represents a
    different class or serogroup, and the rows contain the protein accessions
    that were used as features during model training.

2.  **Aligning** the BLAST scores to the expected feature order, filling
    missing proteins with ``0.0`` (absence = no hit).

Functions
---------
get_features_from_list
    Read a named column from a TSV and return its values as a list.
format_features_for_model
    Build a single-row DataFrame aligned to the expected feature columns.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


def get_features_from_list(file_path: str, column_name: str) -> List[str]:
    """Read protein accessions from a named column in a TSV file.

    This is used to retrieve the exact set of proteins that a particular
    model expects as input features.  The TSV file typically has one column
    per class or serogroup, and each row lists a protein accession.

    Parameters
    ----------
    file_path : str
        Path to the TSV file (tab-separated).
    column_name : str
        Name of the column to extract (e.g. ``"Class 1"`` or ``"Ballum"``).

    Returns
    -------
    list[str]
        Ordered list of protein accession strings.  Returns an empty list
        if the file does not exist, the column is not found, or an I/O
        error occurs.

    Examples
    --------
    >>> features = get_features_from_list("data/seroclasses_to_train.tsv", "Class 1")
    >>> len(features)
    42
    """
    if not os.path.exists(file_path):
        print(f"  [WARN] Feature file not found: {file_path}")
        return []

    try:
        df = pd.read_csv(file_path, sep="\t")
        df.columns = df.columns.str.strip()

        if column_name not in df.columns:
            print(
                f"  [WARN] Column '{column_name}' not found in "
                f"{os.path.basename(file_path)}. "
                f"Available: {list(df.columns)}"
            )
            return []

        return df[column_name].dropna().astype(str).str.strip().tolist()

    except Exception as exc:
        print(f"  [ERROR] Failed to read {file_path}: {exc}")
        return []


def format_features_for_model(
    protein_scores: Dict[str, float],
    expected_features: List[str],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame aligned to *expected_features*.

    Proteins present in *protein_scores* but absent from *expected_features*
    are silently ignored (they are irrelevant to the model).  Proteins listed
    in *expected_features* but absent from *protein_scores* are filled with
    ``0.0``, indicating that no BLAST hit was found.

    Parameters
    ----------
    protein_scores : dict[str, float]
        Mapping returned by :func:`blast_runner.run_tblastn`.
    expected_features : list[str]
        Ordered list of protein accessions expected by the model (as returned
        by :func:`get_features_from_list`).
    verbose : bool, optional
        If ``True`` (default), print a debug line with match statistics.

    Returns
    -------
    pandas.DataFrame
        A single-row DataFrame with columns in the exact order of
        *expected_features*.
    """
    feature_dict = {feat: 0.0 for feat in expected_features}
    matched = 0

    for protein, score in protein_scores.items():
        if protein in feature_dict:
            feature_dict[protein] = score
            matched += 1

    if verbose:
        print(
            f"    → Features expected: {len(expected_features)} | "
            f"Matched from BLAST: {matched}"
        )

    return pd.DataFrame([feature_dict], columns=expected_features)
