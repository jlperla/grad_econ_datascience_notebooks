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
