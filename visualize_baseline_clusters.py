"""
Visualization of Baseline GMM Clustering Results
-------------------------------------------------
Produces two publication-quality charts:
  1. cluster_vs_position.png  – annotated heatmap of cluster × role distribution
  2. cluster_average_heatmaps.png – average offensive presence heatmap per cluster
"""

import json
import math
from pathlib import Path
from typing import Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from mplsoccer import Pitch

PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "baseline_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_paths() -> Tuple[Path, Path, Path]:
    """Return: clusters_csv, profiles_pkl, final_data_dir."""
    with open(CREDS_FILE) as f:
        cfg = json.load(f)

    final_data_dir = Path(cfg["final_data"])
    profiles_dir   = final_data_dir.parent / "player_spatial_profiles"
    clusters_csv   = OUTPUT_DIR / "baseline_gmm_clusters.csv"
    profiles_pkl   = profiles_dir / "processed_player_profiles.pkl"

    for p in (clusters_csv, profiles_pkl, final_data_dir):
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    return clusters_csv, profiles_pkl, final_data_dir


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate dominant role_name per player from the final parquet files
# ─────────────────────────────────────────────────────────────────────────────

def build_player_role_map(final_data_dir: Path) -> pd.Series:
    """
    Scan all final match parquets and return a Series:
        player_id  ->  dominant role_name (SkillCorner positional role)
    """
    print("Extracting dominant role_name per player from final match files …")

    # We only need two lightweight columns; skip heavy event/coordinate cols
    role_accumulator: Dict[int, Dict[str, int]] = {}

    for path in sorted(final_data_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["player_id", "role_name"])
        df = df.dropna(subset=["player_id", "role_name"])
        for pid, sub in df.groupby("player_id"):
            pid = int(pid)
            counts = sub["role_name"].value_counts()
            if pid not in role_accumulator:
                role_accumulator[pid] = {}
            for role, cnt in counts.items():
                role_accumulator[pid][role] = role_accumulator[pid].get(role, 0) + int(cnt)

    dominant = {
        pid: max(roles, key=roles.get)
        for pid, roles in role_accumulator.items()
    }
    role_map = pd.Series(dominant, name="position")
    role_map.index.name = "player_id"
    print(f"  Roles found for {len(role_map)} players. Unique roles: {sorted(role_map.unique())}")
    return role_map


# ─────────────────────────────────────────────────────────────────────────────
# Pitch grid reconstruction (mirrors build_player_spatial_profiles.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_grid_edges(final_data_dir: Path, n_bins: int = 50
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute the x/y histogram edges that were used when building the pkl."""
    sample_path = sorted(final_data_dir.glob("*.parquet"))[0]
    df = pd.read_parquet(sample_path, columns=["x", "y"]).dropna()
    sub = df.sample(n=min(len(df), 200_000), random_state=42)

    x_edges = np.linspace(float(sub["x"].min()), float(sub["x"].max()), n_bins + 1, dtype=np.float32)
    y_edges = np.linspace(float(sub["y"].min()), float(sub["y"].max()), n_bins + 1, dtype=np.float32)
    return x_edges, y_edges


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 1: Cluster × Role heatmap
# ─────────────────────────────────────────────────────────────────────────────

# Ordered from defensive to offensive for a natural reading order on the heatmap
ROLE_ORDER = [
    "Goalkeeper",
    "Center Back", "Left Center Back", "Right Center Back",
    "Left Back", "Right Back",
    "Left Wing Back", "Right Wing Back",
    "Defensive Midfield", "Left Defensive Midfield", "Right Defensive Midfield",
    "Left Midfield", "Right Midfield",
    "Attacking Midfield",
    "Left Winger", "Right Winger",
    "Left Forward", "Center Forward", "Right Forward",
]


def plot_cluster_vs_position(merged_df: pd.DataFrame) -> None:
    present_roles = [r for r in ROLE_ORDER if r in merged_df["position"].unique()]
    extra_roles   = [r for r in merged_df["position"].unique() if r not in ROLE_ORDER]
    role_order    = present_roles + sorted(extra_roles)

    n_clusters = sorted(merged_df["primary_cluster"].unique())
    ct = pd.crosstab(merged_df["primary_cluster"], merged_df["position"])
    ct = ct.reindex(columns=role_order, fill_value=0)
    ct = ct.reindex(index=n_clusters, fill_value=0)

    fig_w = max(14, len(role_order) * 0.85)
    fig_h = max(5,  len(n_clusters) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=30,
        linewidths=0.5,
        linecolor="#cccccc",
        cbar_kws={"label": "Player count", "shrink": 0.7},
        ax=ax,
    )

    ax.set_title("Baseline GMM – Cluster vs Actual Tactical Role\n(raw player counts, colour scale capped at 30)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Tactical Role (SkillCorner)", fontsize=10)
    ax.set_ylabel("Primary Cluster", fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "cluster_vs_position.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 2: Average spatial presence heatmap per cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_average_heatmaps(
    merged_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    clusters = sorted(merged_df["primary_cluster"].unique())
    n_clusters = len(clusters)

    n_cols = 3
    n_rows = math.ceil(n_clusters / n_cols)

    # Derive pitch bounds from our empirical edges
    x_min, x_max = float(x_edges[0]), float(x_edges[-1])
    y_min, y_max = float(y_edges[0]), float(y_edges[-1])

    # Pitch is centered at origin; standard half-lengths are used for mplsoccer
    pitch_length = round(x_max - x_min)   # approx metres
    pitch_width  = round(y_max - y_min)

    # Merge cluster assignment into the profiles df
    profiles_with_cluster = profiles_df.merge(
        merged_df[["player_id", "primary_cluster"]], on="player_id", how="inner"
    )

    # mplsoccer pitch: dark sports analytics aesthetic
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="#0d1117",
        line_color="#3a86ff",
        linewidth=1.2,
        goal_type="box",
    )

    fig, axes = pitch.draw(nrows=n_rows, ncols=n_cols,
                           figsize=(7 * n_cols, 5.5 * n_rows))
    fig.patch.set_facecolor("#0d1117")
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    cmap = matplotlib.colormaps["hot"].copy()
    cmap.set_under(alpha=0)   # transparent cells with zero presence

    for i, cluster_id in enumerate(clusters):
        ax = axes_flat[i]

        # Isolate players in this cluster
        cluster_players = profiles_with_cluster[
            profiles_with_cluster["primary_cluster"] == cluster_id
        ]
        n_players = len(cluster_players)

        if n_players == 0:
            ax.set_title(f"Cluster {cluster_id}\n(no players)", color="white", fontsize=11)
            continue

        # Stack Layer 0 (Presence) tensors and average
        tensors = np.stack(cluster_players["spatial_tensor"].values)  # (n, 5, 50, 50)
        avg_presence = tensors[:, 0, :, :].mean(axis=0)               # (50, 50)

        # Smooth very slightly for readability (simple gaussian-like)
        from scipy.ndimage import gaussian_filter
        avg_presence_smoothed = gaussian_filter(avg_presence, sigma=1.0)

        # Normalise to [0, 1] for colour mapping
        vmax = avg_presence_smoothed.max()
        if vmax > 0:
            avg_presence_smoothed = avg_presence_smoothed / vmax

        # pcolormesh needs (n_x_bins+1, n_y_bins+1) edges – we have those
        # Transpose because: histogram2d output is (x, y) but imshow/pcolormesh is (row=y, col=x)
        im = ax.pcolormesh(
            x_edges,  # x edges (horizontal)
            y_edges,  # y edges (vertical)
            avg_presence_smoothed.T,
            cmap=cmap,
            vmin=1e-4,    # hides near-zero cells
            vmax=1.0,
            shading="flat",
            alpha=0.85,
            zorder=2,
        )

        # Majority role label for this cluster
        cluster_role_counts = merged_df[merged_df["primary_cluster"] == cluster_id]["position"].value_counts()
        top_role = cluster_role_counts.index[0] if len(cluster_role_counts) else "Unknown"
        pct_top  = cluster_role_counts.iloc[0] / cluster_role_counts.sum() * 100 if len(cluster_role_counts) else 0

        ax.set_title(
            f"Cluster {cluster_id}  (n={n_players} players)\n"
            f"Top role: {top_role} ({pct_top:.0f}%)",
            color="white", fontsize=10, fontweight="bold", pad=6,
        )

    # Hide unused axes
    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Average Offensive Presence Heatmap per Cluster\n(Layer 0 – all offensive tracking frames)",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "cluster_average_heatmaps.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    clusters_csv, profiles_pkl, final_data_dir = resolve_paths()

    # Load clustering results
    clusters_df  = pd.read_csv(clusters_csv)
    print(f"Loaded {len(clusters_df)} player cluster records.")

    # Load spatial profiles (for spatial_tensor)
    profiles_df = pd.read_pickle(profiles_pkl)
    print(f"Loaded {len(profiles_df)} player spatial profiles.")

    # Rebuild pitch histogram edges
    print("Reconstructing pitch histogram grid edges …")
    x_edges, y_edges = reconstruct_grid_edges(final_data_dir)
    print(f"  x: [{x_edges[0]:.2f}, {x_edges[-1]:.2f}], y: [{y_edges[0]:.2f}, {y_edges[-1]:.2f}]")

    # Extract dominant role per player
    role_map = build_player_role_map(final_data_dir)

    # Merge position onto clusters dataframe
    merged_df = clusters_df.merge(
        role_map.reset_index(), on="player_id", how="left"
    )
    merged_df["position"] = merged_df["position"].fillna("Unknown")
    print(f"Players with position label: {merged_df['position'].notna().sum()} / {len(merged_df)}")

    # ── Viz 1 ──────────────────────────────────────────────────────────────
    print("\n[1/2] Generating cluster × position heatmap …")
    plot_cluster_vs_position(merged_df)

    # ── Viz 2 ──────────────────────────────────────────────────────────────
    print("\n[2/2] Generating per-cluster average spatial heatmaps …")
    plot_cluster_average_heatmaps(merged_df, profiles_df, x_edges, y_edges)

    # ── Terminal summary ───────────────────────────────────────────────────
    n_clusters = merged_df["primary_cluster"].nunique()
    print(f"\n{'═'*52}")
    print(f"  Optimal clusters (baseline GMM): {n_clusters}")
    print(f"{'─'*52}")
    cluster_counts = merged_df["primary_cluster"].value_counts().sort_index()
    for cid, cnt in cluster_counts.items():
        top_roles = merged_df[merged_df["primary_cluster"] == cid]["position"].value_counts().head(2)
        role_str  = ", ".join(f"{r} ({c})" for r, c in top_roles.items())
        print(f"  Cluster {cid}: {cnt:>4} players  |  top roles: {role_str}")
    print(f"{'═'*52}\n")


if __name__ == "__main__":
    main()
