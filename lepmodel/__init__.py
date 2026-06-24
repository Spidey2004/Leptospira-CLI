"""
lepmodel  source package.

This package implements a command-line tool for predicting Leptospira
serogroups from whole-genome FASTA files.  The pipeline is composed of
four modules:

- :mod:`lepmodel .blast_runner`         — BLAST database creation & tBLASTn execution
- :mod:`lepmodel .feature_extraction`   — Transform BLAST scores into ML features
- :mod:`lepmodel .ml_predictor`         — Two-stage hierarchical ML prediction
- :mod:`lepmodel .utils`                — File discovery, workspace management
- :mod:`lepmodel .cli`                  — Command-line interface (entry point)
"""

__version__ = "1.0.0"
