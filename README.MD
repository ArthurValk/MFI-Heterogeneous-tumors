# Heterogeneous Tumors with Fusions and Chemotherapy

Welcome to the codebase for the **2025-2026 UU Mathematics For Industry project** that models the behavior of cancer cells incorporating cell fusions, mutations, and chemotherapy treatments.

## What's in This Repository

- **Notebooks**: Interactive [marimo](https://marimo.io) notebooks for analysis and visualization
  - Monte Carlo parameter sweep simulations and analysis
  - Model exploration and results visualization

- **ODEs**: Core ODE implementations for the heterogeneous tumor model

## Getting Started

### 1. Install Dependencies

Install `uv`:
```bash
pip install uv
```

Create the virtual environment and install all dependencies from `pyproject.toml`:
```bash
uv sync --all-groups
```

Activate the virtual environment:
```bash
.venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
```

For editor configuration, set your Python interpreter to the `.venv` folder.

### 2. Run the Notebooks

With the virtual environment activated, run the marimo notebooks:

```bash
marimo edit notebooks/monte_carlo/monte_carlo_param_sweep.py
```

Replace the notebook name with whichever notebook you'd like to explore. The `marimo edit` command opens the interactive notebook in your browser.

In the notebooks set the 'output_dir' variable to a directory where you want to save the results of the simulations (in the case of `monte_carlo_param_sweep.py`) or the results you want to visualize (`monte_carlo_param_sweep_analysis.py`).

## Project Structure

```
notebooks/
  └── monte_carlo/          # Parameter sweep simulations and analysis
ODEs/
  └── ode.py               # Core ODE model implementations
```