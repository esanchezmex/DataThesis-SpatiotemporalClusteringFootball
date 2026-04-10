# Spatiotemporal Representation Learning in Football: Defining Fluid Tactical Roles to Optimize Expected Goals (xG)

This repository contains the code for a thesis pipeline that learns **spatiotemporal representations** of outfield players from **merged match tracking and event data**, clusters players into **fluid tactical role groups**, and relates **team-level role mixtures** to **expected goals (xG)**. A **scalar baseline** (no convolutional spatial encoder) is included for comparison with an **autoencoder** trained on multi-channel pitch tensors.

---

## Abstract / Executive Summary

The position-based approach to understanding player function has become increasingly less relevant as the game has evolved toward dynamic, movement-oriented, tactical systems. The traditional static models fail to provide a realistic representation of the positional ambiguity inherent in the game today. In turn, the structural ambiguity in modern football creates an additional challenge to the **"squad composition problem"** where clubs are constantly investing large sums of money on players but lack objective frameworks to measure whether those investments will result in spatially aligned lineups that complement each other. To begin to address this void, this research presents a **multi-dimensional framework to empirically identify clusters of tactical roles and to calculate how the composition of these roles within a lineup influences attacking efficiency.** The study utilizes optical tracking data and play-by-play event data from a full professional domestic league season. Those events were transformed into five layer spatial tensors (Presence, Passes, Carries, Goal Threat, Receptions), to profile players based on their complete attacking footprint. **A 16-dimensional spatiotemporal autoencoder** was utilized to compress these high-dimensional tensors into a condensed representation of latent tactical features. Then, a Gaussian Mixture Model (GMM) was applied to the results of the autoencoder to generate empirically based clusters of player roles. Multiple linear regression (MLR) was employed to determine the relationship between the composition of the roles within a given lineup and the expected number of goals per ninety minutes (xG/90). Ultimately, the clustering pipeline developed here resolved many of the positional ambiguities associated with previous studies by developing ten empirical macro-roles; the MLR modeling provided directional evidence that xG/90 is influenced by the configurations and synergies of these spatial profiles. Therefore, this research provides a quantitative framework for converting tactical intuition into a calculable optimization problem.

---

### Model Architecture

<img src="Spatiotemporal Autoencoder Diagram-1.png" alt="Spatiotemporal Autoencoder Flowchart" width="600" />


---

## Privacy and data

- Committed code and generated artifacts are intended to use **non-identifying keys** (for example numeric IDs) rather than player or club names in published outputs. Treat any raw vendor exports as **confidential**.
- **Credentials and local path configuration** live under `creds/` (see `.gitignore`). Do **not** commit API keys, passwords, or full copies of licensed raw data into this repository.
- Raw parquet and tracking inputs are expected **outside** the repo or in paths pointed to by `creds/gdrive_folder.json`.

---

## Repository layout

| Path | Purpose |
|------|--------|
| Root `*.py` | Runnable pipeline and analysis scripts (run from repository root). |
| `data/outputs/` | Regenerated figures, CSVs, and model weights (`baseline_model/`, `autoencoder/`, `clusters/`, `regression/`, etc.). |
| `creds/` | Local-only JSON (e.g. `gdrive_folder.json`). Not tracked by git. |
| `requirements.txt` | Pinned Python dependencies for reproducibility. |
| `LICENSE` | GNU GPL v3 (see file for full text). |

---

## Environment

- Use **Python 3.10+** (the project has been used with 3.14; match your local setup to `requirements.txt` if you hit binary wheel issues).
- Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- The **convolutional autoencoder** (`train_spatial_autoencoder.py`) will use **CUDA** when available via PyTorch; CPU runs are supported but slower.

---

## Configuration: `creds/gdrive_folder.json`

Create `creds/gdrive_folder.json` at the repository root. Scripts resolve paths from this file so **competition- or export-specific filenames** stay out of source control.

**Required keys:**

| Key | Role |
|-----|------|
| `data_folder_path` | Root folder for thesis data (merged outputs, mappings, etc.). |
| `statsbomb_data_folder_path` | Read-only directory containing vendor **events / matches / player-season** parquet exports. |
| `merged_parquets_folder_path` | Directory of per-match merged parquets produced by the merge step (input to `final_preprocessing.py`). |
| `final_data` | Directory where cleaned per-match parquets are written (input to profile building and several analyses). |
| `statsbomb_events_parquet` | Basename or absolute path to the **events** parquet (resolved under `statsbomb_data_folder_path` if relative). |
| `statsbomb_matches_parquet` | Same for **matches** parquet. |
| `statsbomb_player_season_parquet` | Same for **player-season** parquet. |

If any of the `statsbomb_*_parquet` entries are missing or empty, `merge_tracking_events.py` will raise a clear error at import time.

**Note:** `merge_tracking_events.py` may still contain **machine-specific** defaults (for example a local folder for tracking JSON batches). Review the top of that file after cloning on a new machine.

---

## Pipeline: recommended run order

Run each command from the **repository root** after activating the venv.

1. **`merge_tracking_events.py`** — Aligns tracking with event data and writes merged per-match outputs and mapping artifacts under `data_folder_path`. Requires configured raw inputs and paths inside the merge script where not yet moved to JSON.
2. **`final_preprocessing.py`** — Reads merged parquets from `merged_parquets_folder_path`, applies cleaning and consistency rules, writes to `final_data`.
3. **`build_player_spatial_profiles.py`** — Aggregates season-level **5×50×50** spatial tensors and scalars per player; reads `final_data` from `creds/gdrive_folder.json` and writes **`processed_player_profiles.pkl` at the repository root** (see `paths.py`).
4. **`baseline_gmm_clustering.py`** — Gaussian mixture model on **scalar** features only; writes `data/outputs/baseline_model/baseline_gmm_clusters.csv`.
5. **`visualize_baseline_clusters.py`** and **`visualize_baseline_cluster_vs_position.py`** — Figures for baseline clusters (heatmaps and role crosstabs).
6. **`train_spatial_autoencoder.py`** — Trains the spatial autoencoder, sweeps latent size, writes `data/outputs/autoencoder/` (weights, `ml_ready_features_optimal.csv`, tuning plots, t-SNE, etc.).
7. **`autoencoder_gmm_clustering.py`** — Fits a GMM on latent features; writes `data/outputs/autoencoder/autoencoder_gmm_clusters.csv`.
8. **`visualize_autoencoder_cluster_vs_position.py`** — Autoencoder cluster × role visualization.
9. **`tactical_profiler.py`** — Cluster-level tactical heatmaps (spatial tensor layers) under `data/outputs/clusters/`.
10. **`team_cluster_composition.py`** — Team composition summaries from cluster probabilities.
11. **`cluster_stats_table.py`** — Per-cluster event-style statistics table (CSV under `data/outputs/clusters/`).
12. **`render_cluster_stats_table_png.py`** — Renders the stats table CSV to a PNG (optional if you only need the CSV).
13. **`role_mix_xg_regression.py`** — Team role mixture vs match xG; outputs under `data/outputs/regression/`.
14. **`plot_role_coefficients.py`** — Coefficient forest-style plots from saved regression summaries.

**If artifacts already exist**, you can rerun only the visualization or regression steps that read from `data/outputs/` and `creds`-resolved `final_data`, as long as upstream files are unchanged.

---

## Main outputs (first places to look)

- **`data/outputs/baseline_model/`** — Baseline GMM assignments and diagnostic figures.  
- **`data/outputs/autoencoder/`** — Latent features, tuned weights, t-SNE / reconstruction figures, cluster vs position plots.  
- **`data/outputs/clusters/`** — Tactical profile figures, team composition, cluster statistics.  
- **`data/outputs/regression/`** — OLS-style summaries, team-level caches, coefficient plots.

---

## Optional scripts

- **`quick_pca_preview.py`** — Quick 2D PCA of latent vectors (uses trained autoencoder weights).  
- **`print_autoencoder_cluster_crosstab.py`** — Console crosstab / GMM sanity check from latent CSV.

These are **not** required for the end-to-end story if you rely on the primary scripts above.

## License

This project is licensed under the **GNU General Public License v3.0** — see [`LICENSE`](LICENSE).

---

## Citation

If you reuse this repository, please cite the thesis:

> **Spatiotemporal Representation Learning in Football: Defining Fluid Tactical Roles to Optimize Expected Goals (xG)**

Esteban Sanchez Perezconde, IE University, 2026.
