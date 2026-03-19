from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"
AUTOENCODER_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder"
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_players_df() -> pd.DataFrame:
    with open(CREDS_FILE) as f:
        cfg = json.load(f)

    data_root = Path(cfg["data_folder_path"])
    players_csv = data_root / "skillcorner" / "players_df.csv"
    if not players_csv.exists():
        raise FileNotFoundError(f"players_df.csv not found at expected location: {players_csv}")

    players_df = pd.read_csv(players_csv)

    # Keep only non-identifiable keys we need.
    cols = [c for c in players_df.columns if c in {"player_id", "team_id"}]
    if "player_id" not in cols or "team_id" not in cols:
        raise ValueError("players_df.csv must contain 'player_id' and 'team_id' columns.")

    return players_df[cols].copy()


def main() -> None:
    clusters_csv = AUTOENCODER_DIR / "autoencoder_gmm_clusters.csv"
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Missing autoencoder clustering CSV: {clusters_csv}")

    clusters_df = pd.read_csv(clusters_csv)
    if "player_id" not in clusters_df.columns:
        raise ValueError("autoencoder_gmm_clusters.csv must contain 'player_id'.")

    prob_cols = [c for c in clusters_df.columns if c.startswith("prob_cluster_")]
    if not prob_cols:
        raise ValueError("autoencoder_gmm_clusters.csv must contain prob_cluster_* columns.")

    players_df = load_players_df()

    merged = clusters_df.merge(players_df, on="player_id", how="inner")
    if merged.empty:
        raise ValueError("Merge of clusters with players_df is empty; check player_id consistency.")

    # Aggregate to team-level mixture (mean of probabilities per team).
    team_probs = merged.groupby("team_id")[prob_cols].mean()

    # Replace raw team_id with anonymised ordinal labels: Team 1, Team 2, ...
    team_ids = team_probs.index.to_list()
    label_map = {tid: f"Team {i+1}" for i, tid in enumerate(team_ids)}
    team_probs.index = [label_map[tid] for tid in team_ids]

    # Sort teams by share of the highest-index cluster for a stable ordering.
    sort_col = prob_cols[-1]
    team_probs = team_probs.sort_values(by=sort_col, ascending=False)

    # Plot stacked bar chart of team compositions.
    ax = team_probs.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 6),
        colormap="tab10",
        width=0.9,
    )
    ax.set_ylabel("Cluster share (0–1)")
    ax.set_xlabel("Team")
    ax.set_title("Team tactical composition — autoencoder GMM clusters")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_path = OUT_DIR / "team_cluster_composition.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved team cluster composition plot to: {out_path}")


if __name__ == "__main__":
    main()

