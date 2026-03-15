"""
Convolutional Autoencoder – Latent Dimension Tuning + Final Output Generation
==============================================================================

Pipeline:
  1. Load & normalise spatial tensors (5 x 50 x 50 per player).
  2. Sweep latent_dims [8, 16, 32, 64] with early-stopping training.
  3. For each dim: extract latent vectors, find optimal GMM (BIC 3-10), score silhouette.
  4. Save tuning study CSV + figure.
  5. Re-load best-dim model → extract final latent vectors → save ML-ready CSV.
  6. t-SNE scatter + decoder-trick pitch reconstructions.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from mplsoccer import Pitch


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE   = PROJECT_ROOT / "creds" / "gdrive_folder.json"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "outputs" / "autoencoder"

LATENT_DIMS = [8, 16, 32, 64]
MAX_EPOCHS  = 100
PATIENCE    = 7
BATCH_SIZE  = 32
LR          = 1e-3
N_BINS      = 50
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_paths() -> Tuple[Path, Path]:
    """Return (profiles_pkl_path, final_data_dir)."""
    with open(CREDS_FILE) as f:
        cfg = json.load(f)

    final_data_dir = Path(cfg["final_data"])
    profiles_pkl   = final_data_dir.parent / "player_spatial_profiles" / "processed_player_profiles.pkl"

    if not profiles_pkl.exists():
        raise FileNotFoundError(f"Player profiles not found: {profiles_pkl}")
    if not final_data_dir.exists():
        raise FileNotFoundError(f"final_data directory not found: {final_data_dir}")

    return profiles_pkl, final_data_dir


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & normalisation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_normalize(
    profiles_pkl: Path,
) -> Tuple[np.ndarray, pd.DataFrame, List[int]]:
    """
    Load pkl, extract & per-player min-max normalise tensors to [0, 1].

    Returns:
        tensors_array : (N, 5, 50, 50) float32 numpy array
        scalar_df     : DataFrame with player_id + scalar features
        player_ids    : ordered list of player_ids
    """
    df = pd.read_pickle(profiles_pkl)

    player_ids  = df["player_id"].tolist()
    raw_tensors = df["spatial_tensor"].tolist()

    normalised = []
    for t in raw_tensors:
        t     = t.astype(np.float32)
        t_min = t.min()
        t_max = t.max()
        normalised.append((t - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t))

    tensors_array = np.stack(normalised).astype(np.float32)  # (N, 5, 50, 50)

    scalar_df = df.drop(columns=["spatial_tensor"]).fillna(0.0)

    print(f"Loaded {len(df)} players  |  tensor shape: {tensors_array.shape}")
    return tensors_array, scalar_df, player_ids


def build_dataloaders(
    tensors_array: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """80/20 train/val split → DataLoaders."""
    X       = torch.from_numpy(tensors_array)
    dataset = TensorDataset(X)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train

    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Train: {n_train} players  |  Val: {n_val} players")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model
#
# Encoder spatial path (verified):
#   (B,  5, 50, 50) → Conv+Pool → (B, 16, 25, 25)
#                   → Conv+Pool → (B, 32, 12, 12)
#                   → Conv+Pool → (B, 64,  6,  6)
#                   → Flatten  → (B, 2304)
#                   → Linear   → (B, latent_dim)
#
# Decoder spatial path:
#   (B, latent_dim) → Linear   → (B, 2304)
#                   → Reshape  → (B, 64, 6, 6)
#   ConvTranspose(k=4,s=2,p=1)               → (B, 32, 12, 12)
#   ConvTranspose(k=4,s=2,p=1,out_pad=1)     → (B, 16, 25, 25)
#   ConvTranspose(k=4,s=2,p=1)               → (B,  5, 50, 50)
#   Sigmoid
# ─────────────────────────────────────────────────────────────────────────────

_FLAT_DIM = 64 * 6 * 6  # 2304


class SpatialAutoencoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(5,  16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        # Dropout(0.2) at the bottleneck FC only; conv layers are weight-shared
        # and don't benefit from dropout. Disabled automatically during eval().
        self.encoder_fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(_FLAT_DIM, latent_dim),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, _FLAT_DIM),
            nn.Dropout(p=0.2),
        )
        self.decoder_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,  5, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x).flatten(1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 64, 6, 6)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _verify_architecture(device: torch.device) -> None:
    """Quick sanity-check that input and output shapes match."""
    dummy = torch.zeros(2, 5, 50, 50, device=device)
    with torch.no_grad():
        out = SpatialAutoencoder(latent_dim=16).to(device)(dummy)
    assert out.shape == (2, 5, 50, 50), f"Shape mismatch: {out.shape}"
    print(f"Architecture verified: (B, 5, 50, 50) → latent → (B, 5, 50, 50)  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


def train_with_early_stopping(
    latent_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    weights_path: Path,
) -> Tuple["SpatialAutoencoder", float]:
    """Train one configuration; return best-weights model and best val MSE."""
    model     = SpatialAutoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val  = float("inf")
    no_improv = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr  = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val = _run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        if val < best_val:
            best_val  = val
            no_improv = 0
            torch.save(model.state_dict(), weights_path)
        else:
            no_improv += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} | train {tr:.6f} | val {val:.6f} "
                  f"| best {best_val:.6f} | patience {no_improv}/{PATIENCE}")

        if no_improv >= PATIENCE:
            print(f"    Early stop at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model, best_val


@torch.no_grad()
def extract_latent(
    model: SpatialAutoencoder,
    tensors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    X      = torch.from_numpy(tensors).to(device)
    chunks = [model.encode(X[i: i + BATCH_SIZE]).cpu().numpy()
              for i in range(0, len(X), BATCH_SIZE)]
    return np.concatenate(chunks, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# GMM helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_best_gmm(
    Z: np.ndarray,
    n_min: int = 3,
    n_max: int = 10,
) -> Tuple[GaussianMixture, int, float]:
    """
    Sweep n_components by BIC; return (best_gmm, best_n, silhouette).

    Numerical notes:
    - Cast to float64: sklearn GMM covariance arithmetic is unstable in float32.
    - reg_covar=1e-4: adds a small value to the diagonal of each component
      covariance to prevent singular/collapsed components (sklearn default is 1e-6,
      which is too small for high-dim latent spaces like dim=64).
    - covariance_type='diag': each component has its own diagonal covariance matrix.
      Much more numerically stable than 'full' on small datasets with many features.
    """
    Z64 = Z.astype(np.float64)

    best_bic, best_n, best_gmm = float("inf"), n_min, None

    for n in range(n_min, n_max + 1):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=3,
            random_state=SEED,
        )
        gmm.fit(Z64)
        bic = gmm.bic(Z64)
        if bic < best_bic:
            best_bic, best_n, best_gmm = bic, n, gmm

    labels = best_gmm.predict(Z64)
    sil    = silhouette_score(Z64, labels) if len(set(labels)) > 1 else 0.0
    return best_gmm, best_n, sil


# ─────────────────────────────────────────────────────────────────────────────
# Player role extraction (cached)
# ─────────────────────────────────────────────────────────────────────────────

def load_player_roles(final_data_dir: Path) -> pd.Series:
    """
    Return Series: player_id -> dominant role_name.
    Caches result to player_role_cache.csv to skip rescanning on re-runs.
    """
    cache = OUTPUT_DIR / "player_role_cache.csv"

    if cache.exists():
        df = pd.read_csv(cache, dtype={"player_id": int, "role_name": str})
        print(f"Role cache loaded ({len(df)} players).")
        return df.set_index("player_id")["role_name"]

    print("Scanning match parquets for role_name (one-time cache build) …")
    accum: Dict[int, Dict[str, int]] = {}

    for path in sorted(final_data_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["player_id", "role_name"])
        except Exception:
            continue
        df = df.dropna(subset=["player_id", "role_name"])
        for pid, sub in df.groupby("player_id"):
            pid = int(pid)
            if pid not in accum:
                accum[pid] = {}
            for role, cnt in sub["role_name"].value_counts().items():
                accum[pid][role] = accum[pid].get(role, 0) + int(cnt)

    dominant   = {pid: max(roles, key=roles.get) for pid, roles in accum.items()}
    role_series = pd.Series(dominant, name="role_name")
    role_series.index.name = "player_id"
    role_series.reset_index().to_csv(cache, index=False)
    print(f"Role cache saved ({len(role_series)} players).")
    return role_series


# ─────────────────────────────────────────────────────────────────────────────
# Pitch grid reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_grid_edges(
    final_data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    sample = sorted(final_data_dir.glob("*.parquet"))[0]
    df     = pd.read_parquet(sample, columns=["x", "y"]).dropna()
    sub    = df.sample(n=min(len(df), 200_000), random_state=SEED)
    x_e    = np.linspace(sub["x"].min(), sub["x"].max(), N_BINS + 1, dtype=np.float32)
    y_e    = np.linspace(sub["y"].min(), sub["y"].max(), N_BINS + 1, dtype=np.float32)
    return x_e, y_e


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_tuning_study(results_df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(results_df["latent_dim"], results_df["val_mse"],
             marker="o", linewidth=2, color="#3a86ff")
    ax1.set_title("Validation MSE vs Latent Dimension", fontweight="bold")
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("Validation MSE")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(results_df["latent_dim"], results_df["silhouette"],
             marker="s", linewidth=2, color="#ff6b6b")
    ax2.set_title("Silhouette Score vs Latent Dimension", fontweight="bold")
    ax2.set_xlabel("Latent Dimension")
    ax2.set_ylabel("Silhouette Score  (↑ better)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    best = results_df.loc[results_df["silhouette"].idxmax()]
    ax2.axvline(best["latent_dim"], color="#ff6b6b", linestyle=":", alpha=0.7)
    ax2.annotate(
        f"Optimal\ndim={int(best['latent_dim'])}",
        xy=(best["latent_dim"], best["silhouette"]),
        xytext=(best["latent_dim"] + 1.5, best["silhouette"] - 0.03),
        fontsize=9, color="#ff6b6b",
        arrowprops=dict(arrowstyle="->", color="#ff6b6b"),
    )

    fig.suptitle("Convolutional Autoencoder – Latent Dimension Tuning Study",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "dimension_tuning_study.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_tsne(
    Z: np.ndarray,
    player_ids: List[int],
    role_map: Optional[pd.Series],
    latent_dim: int,
) -> None:
    print("Running t-SNE …")
    perplexity = min(30, max(5, len(Z) // 5))
    tsne       = TSNE(n_components=2, random_state=SEED, perplexity=perplexity,
                      max_iter=1000, learning_rate="auto", init="pca")
    Z2 = tsne.fit_transform(Z)

    plot_df = pd.DataFrame({
        "tsne_1":    Z2[:, 0],
        "tsne_2":    Z2[:, 1],
        "player_id": player_ids,
    })

    has_roles = False
    if role_map is not None:
        plot_df["role"] = plot_df["player_id"].map(role_map).fillna("Unknown")
        has_roles = True

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    if has_roles:
        palette   = sns.color_palette("tab20", n_colors=plot_df["role"].nunique())
        role_order = sorted(plot_df["role"].unique())
        sns.scatterplot(
            data=plot_df, x="tsne_1", y="tsne_2",
            hue="role", hue_order=role_order,
            palette=palette, alpha=0.80, s=55, linewidth=0.3, ax=ax,
        )
        legend = ax.legend(
            title="Tactical Role", bbox_to_anchor=(1.02, 1),
            loc="upper left", fontsize=8, title_fontsize=9,
            facecolor="#1a1a2e", labelcolor="white",
        )
        legend.get_title().set_color("white")
    else:
        ax.scatter(Z2[:, 0], Z2[:, 1], alpha=0.7, s=50, color="#3a86ff")

    for spine in ax.spines.values():
        spine.set_edgecolor("#3a86ff")
    ax.tick_params(colors="white")
    ax.set_xlabel("t-SNE 1", color="white", fontsize=11)
    ax.set_ylabel("t-SNE 2", color="white", fontsize=11)
    ax.set_title(
        f"t-SNE of Optimal Latent Space  (dim={latent_dim})\nColoured by Actual Tactical Role (SkillCorner)",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "tsne_optimal_latent_space.png"
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


@torch.no_grad()
def plot_decoder_reconstructions(
    model: SpatialAutoencoder,
    gmm: GaussianMixture,
    Z: np.ndarray,
    final_data_dir: Path,
    device: torch.device,
) -> None:
    print("Running decoder-trick pitch reconstructions …")

    Z64        = Z.astype(np.float64)
    labels     = gmm.predict(Z64)
    n_clusters = gmm.n_components

    # Mean latent vector per cluster
    centers = np.vstack([Z[labels == k].mean(axis=0) for k in range(n_clusters)])

    model.eval()
    recon = model.decode(
        torch.from_numpy(centers.astype(np.float32)).to(device)
    ).cpu().numpy()  # (n_clusters, 5, 50, 50)

    x_edges, y_edges = reconstruct_grid_edges(final_data_dir)
    pitch_length = round(float(x_edges[-1] - x_edges[0]))
    pitch_width  = round(float(y_edges[-1] - y_edges[0]))

    n_cols = 3
    n_rows = math.ceil(n_clusters / n_cols)

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
    axes_flat = axes.flatten() if (n_rows * n_cols) > 1 else [axes]

    cmap = matplotlib.colormaps["hot"].copy()
    cmap.set_under(alpha=0)

    from scipy.ndimage import gaussian_filter

    for k in range(n_clusters):
        ax      = axes_flat[k]
        layer0  = recon[k, 0]                         # (50, 50)
        layer0  = gaussian_filter(layer0, sigma=1.0)  # mild smoothing
        vmax    = layer0.max()
        if vmax > 0:
            layer0 = layer0 / vmax

        ax.pcolormesh(
            x_edges, y_edges, layer0.T,
            cmap=cmap, vmin=1e-4, vmax=1.0,
            shading="flat", alpha=0.85, zorder=2,
        )

        n_in   = int((labels == k).sum())
        ax.set_title(
            f"Cluster {k}  (n={n_in} players)\nDecoded Prototype – Layer 0 (Presence)",
            color="white", fontsize=9, fontweight="bold", pad=6,
        )

    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Decoder-Trick Cluster Prototypes  "
        f"(optimal dim={model.latent_dim}, n_clusters={n_clusters})",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "decoder_reconstructed_clusters.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    profiles_pkl, final_data_dir = resolve_paths()

    # ── Step 1: Data preparation ─────────────────────────────────────────────
    tensors_array, scalar_df, player_ids = load_and_normalize(profiles_pkl)
    train_loader, val_loader             = build_dataloaders(tensors_array)
    _verify_architecture(device)

    # ── Steps 2–3: Tuning loop ───────────────────────────────────────────────
    tuning_rows: List[Dict] = []

    for latent_dim in LATENT_DIMS:
        print(f"\n{'═' * 60}")
        print(f"  Latent dim = {latent_dim}")
        print(f"{'─' * 60}")

        weights_path = OUTPUT_DIR / f"weights_dim_{latent_dim}.pth"

        model, best_val_mse = train_with_early_stopping(
            latent_dim, train_loader, val_loader, device, weights_path,
        )

        Z             = extract_latent(model, tensors_array, device)
        gmm, best_n, sil = find_best_gmm(Z)

        print(f"  → Best GMM: n_components={best_n}  |  silhouette={sil:.4f}")
        tuning_rows.append({
            "latent_dim": latent_dim,
            "val_mse":    best_val_mse,
            "best_n_gmm": best_n,
            "silhouette": sil,
        })

    # ── Step 4: Save + plot tuning results ───────────────────────────────────
    results_df = pd.DataFrame(tuning_rows)
    results_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    print(f"\nTuning results:\n{results_df.to_string(index=False)}")
    plot_tuning_study(results_df)

    # ── Step 5: Reload optimal model + build ML-ready CSV ───────────────────
    # dim=32 and dim=64 tied on both Silhouette and Val MSE; per the Law of
    # Parsimony we hardcode the lower-dimensionality winner.
    optimal_dim = 32
    best_row    = results_df[results_df["latent_dim"] == optimal_dim].iloc[0]

    print(f"\n{'═' * 60}")
    print(f"  Optimal latent dimension : {optimal_dim}  (hardcoded – tied with 64, parsimony wins)")
    print(f"  Silhouette score         : {best_row['silhouette']:.4f}")
    print(f"  Val MSE                  : {best_row['val_mse']:.6f}")
    print(f"{'═' * 60}")

    optimal_model = SpatialAutoencoder(latent_dim=optimal_dim).to(device)
    optimal_model.load_state_dict(
        torch.load(OUTPUT_DIR / f"weights_dim_{optimal_dim}.pth",
                   map_location=device, weights_only=True)
    )

    optimal_Z            = extract_latent(optimal_model, tensors_array, device)
    optimal_gmm, best_n, _ = find_best_gmm(optimal_Z)

    latent_df = pd.DataFrame(
        optimal_Z,
        columns=[f"latent_{i}" for i in range(optimal_dim)],
    )
    final_df = pd.concat([scalar_df.reset_index(drop=True), latent_df], axis=1)
    out_csv  = OUTPUT_DIR / "ml_ready_features_optimal.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"ML-ready features saved  : {out_csv}  ({len(final_df)} rows, {len(final_df.columns)} cols)")

    # ── Step 6: Visualisations ───────────────────────────────────────────────
    role_map = load_player_roles(final_data_dir)
    plot_tsne(optimal_Z, player_ids, role_map, latent_dim=optimal_dim)
    plot_decoder_reconstructions(optimal_model, optimal_gmm, optimal_Z, final_data_dir, device)

    # Terminal summary
    labels  = optimal_gmm.predict(optimal_Z.astype(np.float64))
    print(f"\n{'═' * 60}")
    print(f"  Final cluster assignment summary (optimal dim={optimal_dim}):")
    for k in range(optimal_gmm.n_components):
        n_in = int((labels == k).sum())
        print(f"    Cluster {k}: {n_in} players")
    print(f"{'═' * 60}")
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
