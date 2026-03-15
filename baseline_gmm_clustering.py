import json
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"


def load_profiles_path() -> Path:
    """
    Resolve the path to the processed_player_profiles.pkl file using creds.
    """
    if not CREDS_FILE.exists():
        raise FileNotFoundError(f"Missing creds file: {CREDS_FILE}")

    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)

    final_data_dir = Path(cfg["final_data"])
    parent = final_data_dir.parent
    profiles_dir = parent / "player_spatial_profiles"

    profiles_path = profiles_dir / "processed_player_profiles.pkl"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Player profiles file not found: {profiles_path}")

    return profiles_path


def prepare_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, np.ndarray, StandardScaler, List[str]]:
    """
    Keep player_id, drop spatial_tensor, handle NaNs, scale scalar features.
    """
    if "player_id" not in df.columns:
        raise ValueError("Input dataframe must contain 'player_id' column.")

    player_ids = df["player_id"].copy()

    # Drop spatial tensor; we only use scalar features for this baseline
    feature_df = df.drop(columns=["spatial_tensor"], errors="ignore")

    # Remove identifier column from feature matrix
    feature_cols = [c for c in feature_df.columns if c != "player_id"]
    if not feature_cols:
        raise ValueError("No scalar feature columns found after dropping 'spatial_tensor' and 'player_id'.")

    feature_df = feature_df[feature_cols]

    # Handle NaNs
    feature_df = feature_df.fillna(0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)

    return player_ids, feature_df, X_scaled, scaler, feature_cols


def find_optimal_components(
    X_scaled: np.ndarray,
    n_min: int = 3,
    n_max: int = 10,
    random_state: int = 42,
) -> Tuple[int, List[int], List[float]]:
    """
    Fit GMMs for a range of components and compute BIC scores.
    """
    n_values: List[int] = []
    bic_scores: List[float] = []

    for n_components in range(n_min, n_max + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        n_values.append(n_components)
        bic_scores.append(bic)
        print(f"n_components={n_components}, BIC={bic:.2f}")

    # Lower BIC is better
    best_idx = int(np.argmin(bic_scores))
    best_n = n_values[best_idx]
    print(f"\nOptimal number of components (min BIC): {best_n}")

    return best_n, n_values, bic_scores


def plot_bic(n_values: List[int], bic_scores: List[float], output_path: Path) -> None:
    """
    Plot BIC vs number of components and save to file.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, bic_scores, marker="o")
    plt.xlabel("Number of components (n_components)")
    plt.ylabel("BIC score")
    plt.title("GMM model selection via BIC")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    # Resolve input profiles path
    profiles_path = load_profiles_path()
    print(f"Loading player profiles from: {profiles_path}")

    df = pd.read_pickle(profiles_path)

    # Prepare data (Step 1)
    player_ids, feature_df, X_scaled, scaler, feature_cols = prepare_data(df)
    print(f"Loaded {len(df)} players with {len(feature_cols)} scalar features.")

    # Prepare output directory
    output_dir = PROJECT_ROOT / "data" / "outputs" / "baseline_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: BIC-based model selection
    best_n, n_values, bic_scores = find_optimal_components(X_scaled, n_min=3, n_max=10, random_state=42)

    bic_plot_path = output_dir / "bic_score_evaluation.png"
    plot_bic(n_values, bic_scores, bic_plot_path)
    print(f"BIC plot saved to: {bic_plot_path}")

    # Step 3: Fit final GMM
    final_gmm = GaussianMixture(n_components=best_n, random_state=42)
    final_gmm.fit(X_scaled)

    primary_clusters = final_gmm.predict(X_scaled)
    prob_clusters = final_gmm.predict_proba(X_scaled)

    # Step 4: Build output dataframe
    result_df = feature_df.copy()
    result_df.insert(0, "player_id", player_ids.values)
    result_df["primary_cluster"] = primary_clusters

    for k in range(best_n):
        result_df[f"prob_cluster_{k}"] = prob_clusters[:, k]

    output_csv = output_dir / "baseline_gmm_clusters.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"\nBaseline GMM clustering results saved to: {output_csv}")

    # Cluster summary
    cluster_counts = pd.Series(primary_clusters).value_counts().sort_index()
    print("\nCluster assignment summary:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} players")

    print(f"\nOptimal number of clusters (n_components): {best_n}")


if __name__ == "__main__":
    main()

