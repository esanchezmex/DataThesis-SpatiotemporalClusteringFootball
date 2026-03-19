from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parent
AUTOENCODER_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder"


def main() -> None:
    csv_path = AUTOENCODER_DIR / "ml_ready_features_optimal.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing autoencoder features CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "player_id" not in df.columns:
        raise ValueError("Expected 'player_id' column in ml_ready_features_optimal.csv")

    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError("Expected latent_* columns to fit GMM.")

    Z = df[latent_cols].to_numpy(dtype=np.float64)

    best_bic = float("inf")
    best_gmm: GaussianMixture | None = None
    best_n: int | None = None

    for n in range(3, 11):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=3,
            random_state=42,
        )
        gmm.fit(Z)
        bic = gmm.bic(Z)
        if bic < best_bic:
            best_bic, best_gmm, best_n = bic, gmm, n

    if best_gmm is None or best_n is None:
        raise RuntimeError("Failed to select a GMM model.")

    primary_clusters = best_gmm.predict(Z)
    prob_clusters = best_gmm.predict_proba(Z)

    out = pd.DataFrame({"player_id": df["player_id"].astype("int64")})
    out["primary_cluster"] = primary_clusters

    for k in range(best_n):
        out[f"prob_cluster_{k}"] = prob_clusters[:, k]

    output_csv = AUTOENCODER_DIR / "autoencoder_gmm_clusters.csv"
    out.to_csv(output_csv, index=False)
    print(f"Saved autoencoder GMM clustering results to: {output_csv} (n_components={best_n})")


if __name__ == "__main__":
    main()

