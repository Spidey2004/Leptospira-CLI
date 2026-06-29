"""
cli.py — Command-line entry point for lepmodel.

This is the main script that users invoke from the terminal.  It orchestrates
the full pipeline:

    1.  **Discover** input genome FASTA files.
    2.  **Prepare** a clean workspace with sanitised filenames.
    3.  **BLAST** each genome against the protein reference panel.
    4.  **Extract** features from the BLAST results.
    5.  **Predict** class and serogroup with the pre-trained ML models.
    6.  **Output** a summary CSV and print results to the terminal.
    7.  **Clean up** all temporary files.

Usage
-----
::

    python -m lepmodel -i <input> [-o <output>] [-d <data>] [-m <models>] [-q <queries>]

Run ``python -m lepmodel --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import pandas as pd

from .blast_runner import create_blast_db, run_tblastn
from .ml_predictor import run_ml_pipeline
from .utils import cleanup_workspace, discover_genomes, prepare_workspace


# ── Argument parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="lepmodel",
        description=(
            "Predict the serogroup of Leptospira genomes using a two-stage "
            "hierarchical ML pipeline (Class → Serogroup) built on tBLASTn "
            "protein profiles."
        ),
        epilog=(
            "Example:\n"
            "  python -m lepmodel -i genomes/ -o results/\n\n"
            "BLAST+ (makeblastdb, tblastn) must be installed and on your PATH."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=False,
        default="./data/input",
        metavar="PATH",
        help=(
            "Path to a single genome FASTA file or a directory containing "
            "multiple .fasta / .fas / .fna files."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default="./data/output",
        metavar="DIR",
        help="Directory where results will be saved (default: ./output).",
    )
    parser.add_argument(
        "-d", "--data",
        default="./data",
        metavar="DIR",
        help=(
            "Directory containing the training feature TSV files "
            "(seroclasses_to_train.tsv, serogroups_to_train.tsv). "
            "Default: ./data"
        ),
    )
    parser.add_argument(
        "-m", "--models",
        default="./models",
        metavar="DIR",
        help="Directory containing the pre-trained .pkl model files (default: ./models).",
    )
    parser.add_argument(
        "-q", "--queries",
        default="./data/queries",
        metavar="DIR",
        help=(
            "Directory containing the multi-FASTA protein query files used "
            "by tBLASTn (default: ./data/queries).  Must NOT be the same "
            "directory as --input."
        ),
    )
    parser.add_argument(
        "-R", "--recursive",
        action="store_true",
        default=False,
        help=(
            "Search subdirectories recursively for genome FASTA files. "
            "Useful when genomes are organised in per-sample folders."
        ),
    )

    return parser


# ── Main routine ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the full prediction pipeline.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = os.path.abspath(args.output)
    workspace_dir = os.path.join(output_dir, "genomes_workspace")
    model_dir = os.path.abspath(args.models)
    data_dir = os.path.abspath(args.data)
    queries_dir = os.path.abspath(args.queries)

    os.makedirs(output_dir, exist_ok=True)

    # ── Safety check: input genomes and protein queries must not overlap ─
    input_abs = os.path.abspath(args.input)
    if os.path.isdir(input_abs) and os.path.samefile(input_abs, queries_dir):
        print(
            "\n  [ERROR] --input and --queries resolve to the same directory:\n"
            f"           {input_abs}\n"
            "  The genome FASTA files (nucleotide) and the protein query\n"
            "  files for tBLASTn must be in separate directories.\n\n"
            "  Typical layout:\n"
            "    data/input/    → your genome .fasta/.fas/.fna files\n"
            "    data/queries/  → protein reference multi-FASTA files\n"
        )
        sys.exit(1)

    # ── Banner ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  lepmodel — Leptospira Serogroup Prediction CLI")
    print("=" * 60)
    print(f"  Input     : {args.input}")
    print(f"  Recursive : {'Yes' if args.recursive else 'No'}")
    print(f"  Output    : {output_dir}")
    print(f"  Models    : {model_dir}")
    print(f"  Data      : {data_dir}")
    print(f"  Queries   : {queries_dir}")
    print("=" * 60)

    # ── Step 1: Discover genomes ────────────────────────────────────────
    try:
        genome_paths = discover_genomes(args.input, recursive=args.recursive)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)

    print(f"\n  Found {len(genome_paths)} genome(s) to process.\n")

    # ── Step 2: Prepare workspace ───────────────────────────────────────
    mapping = prepare_workspace(genome_paths, workspace_dir)

    # ── Step 3: BLAST each genome ───────────────────────────────────────
    all_scores: Dict[str, Dict[str, float]] = {}
    total_start = time.time()

    for original, workspace_path in mapping:
        genome_name = os.path.basename(workspace_path)
        print(f"{'=' * 50}")
        print(f"  Genome: {genome_name}")
        print(f"{'=' * 50}")

        t0 = time.time()

        # 3a. Create BLAST DB
        print("  Step 1/2 — Creating BLAST database ...")
        db_name = create_blast_db(workspace_path)
        if db_name is None:
            print(f"  [SKIP] Could not create DB for {genome_name}.\n")
            continue

        # 3b. Run tBLASTn
        print("  Step 2/2 — Running tBLASTn ...")
        scores = run_tblastn(queries_dir, db_name, output_dir)
        all_scores[workspace_path] = scores

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  ({len(scores)} proteins scored).\n")

    if not all_scores:
        print("\n  [ERROR] No BLAST results produced. Aborting.")
        cleanup_workspace(workspace_dir)
        sys.exit(1)

    # ── Step 4: ML prediction ───────────────────────────────────────────
    results = run_ml_pipeline(all_scores, model_dir, data_dir)

    # ── Step 5: Save results ────────────────────────────────────────────
    if results:
        summary_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "BATCH_SUMMARY_RESULTS.csv")
        summary_df.to_csv(csv_path, index=False)

        print(f"\n{'=' * 60}")
        print("  RESULTS")
        print(f"{'=' * 60}")
        print(summary_df.to_string(index=False))
        print(f"\n  Summary saved to: {csv_path}")
    else:
        print("\n  No predictions could be made for any genome.")

    # ── Step 6: Cleanup ─────────────────────────────────────────────────
    cleanup_workspace(workspace_dir)

    total_elapsed = time.time() - total_start
    print(f"\n  Total elapsed time: {total_elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
