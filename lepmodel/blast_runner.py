"""
blast_runner.py — BLAST database creation and tBLASTn execution.

This module encapsulates all interactions with the NCBI BLAST+ command-line
suite.  It is responsible for:

    1.  Building a nucleotide BLAST database from a genome FASTA file.
    2.  Running ``tblastn`` searches (protein query → nucleotide subject).
    3.  Parsing the tabular output and computing a normalised alignment score
        per query protein.

The normalised score is defined as::

    score = pident × (alignment_length / query_length)

Only the **best hit** (highest score) per query protein is retained.

Dependencies
------------
- NCBI BLAST+ must be installed and available on ``$PATH``
  (``makeblastdb`` and ``tblastn``).
- ``pandas`` for tabular parsing.

Functions
---------
create_blast_db
    Wraps ``makeblastdb`` to create a nucleotide database.
count_fasta_sequences
    Counts ``>`` header lines in a FASTA file.
run_tblastn
    Runs ``tblastn`` for every query FASTA in a directory and returns
    a dictionary of ``{protein_accession: best_score}``.
"""

from __future__ import annotations

import glob
import os
import subprocess
from typing import Dict

import pandas as pd


# ── BLAST database ──────────────────────────────────────────────────────────

def create_blast_db(genome_fasta: str) -> str | None:
    """Create a nucleotide BLAST database from a genome FASTA file.

    The database files are written next to the input FASTA with the suffix
    ``_db`` (e.g. ``genome.fasta_db.*``).

    Parameters
    ----------
    genome_fasta : str
        Absolute or relative path to the input genome FASTA.

    Returns
    -------
    str or None
        The database prefix (``<genome_fasta>_db``) on success, or ``None``
        if ``makeblastdb`` fails.
    """
    db_out = genome_fasta + "_db"
    print(f"  [BLAST] Creating database for {os.path.basename(genome_fasta)} ...")
    try:
        subprocess.run(
            ["makeblastdb", "-in", genome_fasta, "-dbtype", "nucl", "-out", db_out],
            check=True,
            capture_output=True,
            text=True,
        )
        print("  [BLAST] Database created successfully.")
        return db_out
    except FileNotFoundError:
        print(
            "  [ERROR] 'makeblastdb' not found. "
            "Please install NCBI BLAST+ and ensure it is on your PATH."
        )
        return None
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] makeblastdb failed: {exc.stderr}")
        return None


# ── Helpers ─────────────────────────────────────────────────────────────────

def count_fasta_sequences(fasta_path: str) -> int:
    """Count the number of sequences (``>`` headers) in a FASTA file.

    Parameters
    ----------
    fasta_path : str
        Path to a FASTA file.

    Returns
    -------
    int
        Number of sequences found.
    """
    count = 0
    with open(fasta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


# ── tBLASTn ─────────────────────────────────────────────────────────────────

def run_tblastn(
    multifasta_dir: str,
    db_name: str,
    output_dir: str,
) -> Dict[str, float]:
    """Run tBLASTn for every query FASTA found in *multifasta_dir*.

    Each ``.fasta`` / ``.fas`` file in *multifasta_dir* is used as a protein
    query against the nucleotide database *db_name*.  Results are parsed from
    BLAST tabular format (``-outfmt 6``) and a normalised score is computed
    for every query protein::

        score = pident × (alignment_length / query_length)

    Only the best hit per protein is kept.

    Parameters
    ----------
    multifasta_dir : str
        Directory containing one or more multi-FASTA protein query files.
    db_name : str
        BLAST database prefix (as returned by :func:`create_blast_db`).
    output_dir : str
        Directory where the temporary BLAST output file will be written.

    Returns
    -------
    dict[str, float]
        Mapping of ``{protein_accession: best_normalised_score}``.
    """
    temp_out = os.path.join(output_dir, "temp_blast.out")
    protein_scores: Dict[str, float] = {}

    fasta_files = (
        glob.glob(os.path.join(multifasta_dir, "*.fasta"))
        + glob.glob(os.path.join(multifasta_dir, "*.fas"))
    )

    if not fasta_files:
        print(f"  [BLAST] No query FASTA files found in {multifasta_dir}. Skipping.")
        return {}

    total_seqs = sum(count_fasta_sequences(f) for f in fasta_files)
    print(
        f"  [BLAST] Running tblastn — {total_seqs} protein sequence(s) "
        f"from {len(fasta_files)} file(s) ..."
    )

    for fasta_file in fasta_files:
        try:
            cmd = [
                "tblastn",
                "-query", fasta_file,
                "-db", db_name,
                "-outfmt", "6 qseqid sseqid pident length qlen",
                "-out", temp_out,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
                cols = ["qseqid", "sseqid", "pident", "length", "qlen"]
                df = pd.read_csv(temp_out, sep="\t", header=None, names=cols)

                df["score"] = df.apply(
                    lambda row: (
                        row["pident"] * (row["length"] / row["qlen"])
                        if row["qlen"] > 0
                        else 0.0
                    ),
                    axis=1,
                )
                best_hits = df.groupby("qseqid")["score"].max().to_dict()
                protein_scores.update(best_hits)

        except FileNotFoundError:
            print(
                "  [ERROR] 'tblastn' not found. "
                "Please install NCBI BLAST+ and ensure it is on your PATH."
            )
            break
        except subprocess.CalledProcessError as exc:
            print(
                f"  [WARN] tblastn failed for "
                f"{os.path.basename(fasta_file)}: {exc.stderr}"
            )
            continue

    # Clean up the temporary file
    if os.path.exists(temp_out):
        os.remove(temp_out)

    return protein_scores
