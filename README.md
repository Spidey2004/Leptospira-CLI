# LeptoSerogrouper

A command-line tool for predicting **Leptospira serogroups** from whole-genome
FASTA assemblies using a two-stage hierarchical machine-learning pipeline.

## How It Works

```
Genome FASTA  ──►  tBLASTn (protein panel)  ──►  Feature matrix  ──►  Class prediction  ──►  Serogroup prediction
```

1. **BLAST** — Each genome is searched against a curated panel of protein
   sequences with `tBLASTn`.  A normalised alignment score is computed for
   every query protein.
2. **Class prediction** — Four binary Balanced Random Forest (BRF) models
   evaluate the protein profile and assign a phylogenetic class (1–4).
3. **Serogroup prediction** — Class-specific One-vs-Rest BRF models predict
   the final serogroup within the assigned class.

## Installation

All dependencies — including **NCBI BLAST+** — are handled automatically
through Conda/Bioconda.  No manual installation of external tools is required.

### Quick start (2 commands)

```bash

# 1. Create the conda environment (installs Python, BLAST+, and all libraries)
conda env create -f environment.yml

# 2. Activate and register the CLI command
conda activate lepmodel
pip install -e .
```

> **Note:** If you don't have Conda installed, we recommend
> [Miniforge](https://github.com/conda-forge/miniforge#install) — it is
> lightweight and comes pre-configured with the `conda-forge` channel.

### Verify the installation

```bash
lepmodel --help          # CLI entry point
makeblastdb -version     # BLAST+ (installed by Bioconda)
tblastn -version
```

## Usage

```bash
insert your genomes in data/input or in your prefered path

run "lepmodel"

analyse the output file in data/output
```

### Required

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Path to a genome FASTA file **or** a directory of `.fasta` / `.fas` / `.fna` files. |

### Optional

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output`  | `./output`     | Directory where results will be saved. |
| `-d`, `--data`    | `./data`       | Directory with training feature TSVs (`seroclasses_to_train.tsv`, `serogroups_to_train.tsv`). |
| `-m`, `--models`  | `./models`     | Directory with pre-trained `.pkl` model files. |
| `-q`, `--queries` | `./data/queries` | Directory with multi-FASTA protein query files for tBLASTn. |

### Examples

```bash
# Single genome
lepmodel -i genomes/Lai.fasta

# Batch — all genomes in a directory
lepmodel -i genomes/ -o results/

# Custom model and data paths
lepmodel -i genomes/ -m /path/to/models -d /path/to/data
```

## Project Structure

```
Modelo_Leptospira/
├── data/
│   ├── input/                  # Place your genome FASTA files here
│   ├── queries/                # Protein reference multi-FASTAs for tBLASTn
│   └── output/                 # (generated) intermediate outputs
├── models/                     # Pre-trained .pkl model files
├── lepmodel/
│   ├── __init__.py             # Package metadata
│   ├── __main__.py             # python -m lepmodel support
│   ├── blast_runner.py         # BLAST DB creation & tBLASTn execution
│   ├── cli.py                  # CLI entry point (argparse)
│   ├── feature_extraction.py   # BLAST scores → ML feature matrices
│   ├── ml_predictor.py         # Two-stage ML prediction pipeline
│   └── utils.py                # File discovery & workspace management
├── tests/                      # Unit tests
├── environment.yml             # Conda env (Python + BLAST+ + libs)
├── pyproject.toml              # Package metadata & CLI entry point
├── .gitignore
├── LICENSE
└── README.md
```

## Output

The tool produces a CSV file (`BATCH_SUMMARY_RESULTS.csv`) in the output
directory with the following columns:

| Column | Description |
|--------|-------------|
| `Genome` | Filename of the input genome. |
| `Predicted_Class` | Phylogenetic class (1–4). |
| `Predicted_Serogroup` | Predicted serogroup name. |
| `Confidence` | Positive-class probability for the serogroup (%). |

## License

This project is open source. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

> *TODO — Add citation details here.*
