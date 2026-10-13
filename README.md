# Graduate Quantitative Economics and Datascience
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jlperla/grad_econ_datascience_notebooks/HEAD)

This repository contains Jupyter notebooks for the [Graduate Quantitative Economics and Datascience](https://jlperla.github.io/grad_econ_datascience/) course (the first half of ECON526 at UBC)


## Installation
To setup, we strongly suggest using the [uv](https://github.com/astral-sh/uv) package management system.  To install:
   - MacOS or Linux: `curl -sSfL https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | sh`, from any terminal
   - Windows: `iwr https://raw.githubusercontent.com/astral-sh/uv/main/install.ps1 -useb | iex`, from a Windows PowerShell terminal (i.e., Open the Start menu, type Windows PowerShell, select Windows PowerShell, then select Open.)

With `uv` after cloning this repository, create a virtual environment and then sync the dependencies with:
```bash
uv sync
```

For spatial data, image classification, and LangChain examples, install the optional packages with:

```bash
uv sync --extra extras
```

See [here](https://jlperla.github.io/grad_econ_datascience/pages/python_setup.html#alternative-conda-instructions) for more details

## Using Notebooks
Open the lecture `.ipynb` files in VS Code and select the repository's `.venv` interpreter.

See [here](https://jlperla.github.io/grad_econ_datascience/pages/python_setup.html) for more details and instructions on how to use with `conda`.

## Handwriting and Digitization

The [handwriting lecture](slides/handwriting.ipynb) includes saved Toronto,
Paris, and Argentine census predictions. Viewing them needs no model or API key.
Download Toronto scans and reference labels from their original hosts:

```bash
uv run --extra handwriting python utilities/build_toronto_census.py
```

On Apple Silicon, rerun OCR in two terminals:

```bash
uv run --extra handwriting-mac python -m mlx_vlm.server \
  --model datalab-to/chandra-ocr-2 --host 127.0.0.1 --port 8081
uv run --extra handwriting python code/toronto_census_ocr.py \
  --workflow chandra --split test
```

See [Qwen setup and scoring](slides/data/README.md#toronto-1901-local-model-comparison)
and the [other handwriting examples](slides/data/README.md#handwriting).
Paris images and labels are included; Toronto and Argentine full scans load remotely.
