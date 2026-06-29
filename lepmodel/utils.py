"""
utils.py — General-purpose utilities for the Leptospira CLI tool.

This module provides helper functions that do not fit neatly into the BLAST,
feature-extraction, or ML modules.  Current responsibilities include:

- Discovering genome FASTA files from a path (file or directory).
- Sanitising filenames to avoid issues with BLAST+ (spaces, parentheses).
- Preparing a temporary workspace with sanitised copies of input genomes.
- Cleaning up the workspace after a run.

Functions
---------
discover_genomes
    Resolve a user-supplied path into a list of genome FASTA files.
prepare_workspace
    Copy genomes into a clean workspace with safe filenames.
cleanup_workspace
    Remove the workspace directory and all its contents.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import List, Tuple

# File extensions recognised as genome FASTA files.
_GENOME_EXTENSIONS = ("*.fasta", "*.fas", "*.fna")


def discover_genomes(input_path: str, *, recursive: bool = False) -> List[str]:
    """Discover genome FASTA files from *input_path*.

    If *input_path* is a single file it is returned directly.  If it is a
    directory, all files matching the extensions ``.fasta``, ``.fas``, and
    ``.fna`` are collected.

    Parameters
    ----------
    input_path : str
        A file path or a directory path supplied by the user.
    recursive : bool, optional
        If ``True``, search subdirectories recursively for genome files.
        Default is ``False`` (only the top-level directory is scanned).

    Returns
    -------
    list[str]
        Sorted list of absolute paths to genome FASTA files.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    ValueError
        If *input_path* is a directory but no FASTA files are found.
    """
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if os.path.isfile(input_path):
        return [input_path]

    # It is a directory — scan for FASTA files
    genomes: List[str] = []
    if recursive:
        for ext in _GENOME_EXTENSIONS:
            genomes.extend(
                glob.glob(os.path.join(input_path, "**", ext), recursive=True)
            )
    else:
        for ext in _GENOME_EXTENSIONS:
            genomes.extend(glob.glob(os.path.join(input_path, ext)))

    if not genomes:
        mode = "recursively in" if recursive else "in"
        raise ValueError(
            f"No genome FASTA files ({', '.join(_GENOME_EXTENSIONS)}) "
            f"found {mode}: {input_path}"
        )

    return sorted(genomes)


def _sanitise_filename(filename: str) -> str:
    """Remove characters that cause issues with BLAST+ command-line tools.

    Spaces are replaced by underscores and parentheses are stripped.

    Parameters
    ----------
    filename : str
        Original filename (basename only, no directory component).

    Returns
    -------
    str
        Sanitised filename.
    """
    return filename.replace(" ", "_").replace("(", "").replace(")", "")


def prepare_workspace(
    genome_paths: List[str],
    workspace_dir: str,
) -> List[Tuple[str, str]]:
    """Copy genome files into *workspace_dir* with sanitised names.

    Parameters
    ----------
    genome_paths : list[str]
        Original genome file paths (as returned by :func:`discover_genomes`).
    workspace_dir : str
        Directory where sanitised copies will be placed.  Created if it does
        not exist.

    Returns
    -------
    list[tuple[str, str]]
        A list of ``(original_path, workspace_path)`` tuples.  The second
        element is the path that should be used for all subsequent BLAST
        operations.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    mapping: List[Tuple[str, str]] = []
    for genome in genome_paths:
        safe_name = _sanitise_filename(os.path.basename(genome))
        dest = os.path.join(workspace_dir, safe_name)
        shutil.copy2(genome, dest)
        mapping.append((genome, dest))

    return mapping


def cleanup_workspace(workspace_dir: str) -> None:
    """Remove the workspace directory and all temporary BLAST artefacts.

    Parameters
    ----------
    workspace_dir : str
        Path to the workspace created by :func:`prepare_workspace`.
    """
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
        print(f"  [CLEANUP] Removed workspace: {workspace_dir}")
    else:
        print("  [CLEANUP] No workspace to clean up.")
