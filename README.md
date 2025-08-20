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

See [here](https://jlperla.github.io/grad_econ_datascience/pages/setup.html#quick-start) for more details

## Using Notebooks
At that point, you can open the notebooks with:
  1. VS Code and select the `./.venv/bin/python` interpreter if prompted **(recommended)**
  2. Run the notebooks from jupyter with `uv run jupyter lab`.  
    - Alternatively, you can activate the virtual environment with `source myvenv/bin/activate` on macos/linux, or `venv\Scripts\activate.bat` on Windows, and then run `jupyter lab` directly.

See [here](https://jlperla.github.io/grad_econ_datascience/slides/environment.html) for more details and instructions on how to use with `conda`.