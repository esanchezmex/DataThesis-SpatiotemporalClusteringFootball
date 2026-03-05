import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"


@dataclass
class PlayerSeasonProfile:
    """
    Container for a single player's season-long spatial and scalar features.
    """

    player_id: int
    spatial_tensor: np.ndarray  # shape: (5, n_x_bins, n_y_bins)
    # Player-level action counts (season totals)
    passes: int = 0
    carries: int = 0
    goal_threat: int = 0
    receptions: int = 0
    total_actions: int = 0
    # Team-level action counts while associated with this player (season totals)
    team_passes: int = 0
    team_carries: int = 0
    team_goal_threat: int = 0
    team_receptions: int = 0
    team_total_actions: int = 0


def load_paths() -> Tuple[Path, Path]:
    """
    Load input/output directories from creds/gdrive_folder.json.
    """
    if not CREDS_FILE.exists():
        raise FileNotFoundError(f"Missing creds file: {CREDS_FILE}")

    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)

    input_dir = Path(cfg["final_data"])  # strictly read-only
    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT (final_data) directory does not exist: {input_dir}")

    parent = input_dir.parent
    output_dir = parent / "player_spatial_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir


def infer_pitch_bounds(sample_df: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Infer pitch coordinate bounds from a sample dataframe.

    Falls back to StatsBomb-style dimensions if necessary.
    """
    if "x" not in sample_df.columns or "y" not in sample_df.columns:
        raise ValueError("Sample dataframe must contain 'x' and 'y' columns for coordinates.")

    # Use a subset to avoid scanning extremely large files
    sub = sample_df[["x", "y"]].dropna().sample(
        n=min(len(sample_df), 200_000), random_state=42
    ) if len(sample_df) > 200_000 else sample_df[["x", "y"]].dropna()

    x_min, x_max = float(sub["x"].min()), float(sub["x"].max())
    y_min, y_max = float(sub["y"].min()), float(sub["y"].max())

    # Simple heuristics to adjust to common football pitch conventions
    # If values look normalized (0–1), rescale to 0–105, 0–68
    if 0.0 <= x_min >= -1.0 and x_max <= 1.0 and 0.0 <= y_min >= -1.0 and y_max <= 1.0:
        return (0.0, 105.0), (0.0, 68.0)

    # If coordinates roughly 0–120 x 0–80 (StatsBomb event space), clamp to 0–105, 0–68
    if 0.0 <= x_min <= 5.0 and 100.0 <= x_max <= 130.0 and 0.0 <= y_min <= 5.0 and 60.0 <= y_max <= 90.0:
        return (0.0, 105.0), (0.0, 68.0)

    # Otherwise just use empirical min/max
    return (x_min, x_max), (y_min, y_max)


def get_event_type_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return the name of the event type column if present.
    """
    for col in ("event_type", "type"):
        if col in df.columns:
            return col
    return None


def build_action_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build boolean masks for different action categories.
    """
    event_col = get_event_type_column(df)
    if event_col is None:
        # No explicit event column; treat everything as non-event (only presence layer will be non-empty).
        false_mask = np.zeros(len(df), dtype=bool)
        return {
            "pass": false_mask,
            "carry": false_mask,
            "goal_threat": false_mask,
            "reception": false_mask,
        }

    etype = df[event_col].astype(str).str.lower()

    is_pass = etype.eq("pass")
    is_carry = etype.eq("carry") | etype.eq("dribble")
    is_shot = etype.eq("shot")

    # Receptions (StatsBomb-style ball receipt)
    is_reception = etype.str.contains("ball receipt", case=False, na=False)

    # Key pass / assist heuristics
    key_pass_mask = np.zeros(len(df), dtype=bool)
    if "pass_assisted_shot_id" in df.columns:
        key_pass_mask |= df["pass_assisted_shot_id"].notna()
    if "key_pass" in df.columns and df["key_pass"].dtype == bool:
        key_pass_mask |= df["key_pass"]

    is_goal_threat = is_shot | (is_pass & key_pass_mask)

    # Return boolean Series aligned with df's index so we can safely
    # subset both the global dataframe and per-player slices.
    return {
        "pass": is_pass,
        "carry": is_carry,
        "goal_threat": is_goal_threat,
        "reception": is_reception,
    }


def get_offensive_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Mask for offensive frames: rows where the player's team is in possession.
    """
    if "team_in_possession" not in df.columns:
        # Fallback: any frame with a known possession is considered offensive for presence
        return df.get("possession").notna().values

    team_in_possession = df["team_in_possession"]

    if "team" in df.columns:
        # Only count rows where the player's team matches the team in possession
        return (team_in_possession.notna() & (team_in_possession == df["team"])).values

    # Fallback: treat any non-null team_in_possession as offensive
    return team_in_possession.notna().values


def update_histograms_for_match(
    df: pd.DataFrame,
    players: Dict[int, PlayerSeasonProfile],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    """
    Update player-level spatial histograms and action counts for a single match.
    """
    if "player_id" not in df.columns:
        raise ValueError("Input data must contain 'player_id' column.")

    offensive_mask = get_offensive_mask(df)
    action_masks = build_action_masks(df)

    # Team identifiers (for team-level action counts)
    team_col = "event_team" if "event_team" in df.columns else ("team" if "team" in df.columns else None)

    # Per-match team action counts
    team_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if team_col is not None:
        event_teams = df[team_col].astype(str).fillna("UNKNOWN").to_numpy()
    else:
        event_teams = np.array(["UNKNOWN"] * len(df), dtype=object)

    # Update team action counts for this match
    for action_name, mask in action_masks.items():
        if not mask.any():
            continue
        for team_value, count in (
            pd.Series(event_teams[mask.values]).value_counts().items()
        ):
            team_counts[team_value][action_name] += int(count)

    # Also track team total actions (sum of all action types)
    for team_value, counts in team_counts.items():
        counts["total_actions"] = (
            counts.get("pass", 0)
            + counts.get("carry", 0)
            + counts.get("goal_threat", 0)
            + counts.get("reception", 0)
        )

    # Determine each player's predominant team in this match (for influence denominators)
    player_team_map: Dict[int, str] = {}
    if "team" in df.columns:
        team_series = df[["player_id", "team"]].dropna()
        if not team_series.empty:
            for pid, sub in team_series.groupby("player_id"):
                # Use mode to pick the predominant team label
                mode_val = sub["team"].mode()
                if not mode_val.empty:
                    player_team_map[int(pid)] = str(mode_val.iloc[0])

    # Iterate over players in this match
    for pid, player_df in df[offensive_mask].groupby("player_id"):
        pid = int(pid)

        # Ensure player profile exists
        if pid not in players:
            spatial_tensor = np.zeros((5, len(x_edges) - 1, len(y_edges) - 1), dtype=np.float32)
            players[pid] = PlayerSeasonProfile(player_id=pid, spatial_tensor=spatial_tensor)

        profile = players[pid]

        # Presence layer (0): all offensive frames for this player
        hx, _, _ = np.histogram2d(
            player_df["x"].to_numpy(),
            player_df["y"].to_numpy(),
            bins=[x_edges, y_edges],
        )
        profile.spatial_tensor[0] += hx.astype(np.float32)

        # For action-based layers we need masks restricted to this player's rows
        idx = player_df.index

        def _hist_for_mask(global_mask: pd.Series) -> np.ndarray:
            # Align global boolean mask to this player's rows
            local_mask = global_mask.loc[idx].values
            if not local_mask.any():
                return np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float32)
            hx_loc, _, _ = np.histogram2d(
                player_df.loc[local_mask, "x"].to_numpy(),
                player_df.loc[local_mask, "y"].to_numpy(),
                bins=[x_edges, y_edges],
            )
            return hx_loc.astype(np.float32)

        # Layer 1 (Passes)
        h_pass = _hist_for_mask(action_masks["pass"])
        profile.spatial_tensor[1] += h_pass

        # Layer 2 (Carries)
        h_carry = _hist_for_mask(action_masks["carry"])
        profile.spatial_tensor[2] += h_carry

        # Layer 3 (Goal Threat)
        h_gt = _hist_for_mask(action_masks["goal_threat"])
        profile.spatial_tensor[3] += h_gt

        # Layer 4 (Receptions)
        h_rec = _hist_for_mask(action_masks["reception"])
        profile.spatial_tensor[4] += h_rec

        # Update player action counts (season totals)
        passes = int(action_masks["pass"].loc[idx].sum())
        carries = int(action_masks["carry"].loc[idx].sum())
        goal_threat = int(action_masks["goal_threat"].loc[idx].sum())
        receptions = int(action_masks["reception"].loc[idx].sum())

        profile.passes += passes
        profile.carries += carries
        profile.goal_threat += goal_threat
        profile.receptions += receptions

        actions_this_match = passes + carries + goal_threat + receptions
        profile.total_actions += actions_this_match

        # Update team-level denominators for this player (season totals)
        team_label = player_team_map.get(pid)
        if team_label is not None:
            tcounts = team_counts.get(team_label, {})
            profile.team_passes += int(tcounts.get("pass", 0))
            profile.team_carries += int(tcounts.get("carry", 0))
            profile.team_goal_threat += int(tcounts.get("goal_threat", 0))
            profile.team_receptions += int(tcounts.get("reception", 0))
            profile.team_total_actions += int(tcounts.get("total_actions", 0))


def compute_scalar_features(players: Dict[int, PlayerSeasonProfile]) -> pd.DataFrame:
    """
    Compute normalized scalar features for all players and return as a dataframe.
    """
    records: List[Dict] = []

    for pid, profile in players.items():
        # Tendency: player-action normalization
        total_actions = profile.total_actions if profile.total_actions > 0 else 1
        pass_tendency = profile.passes / total_actions
        carry_tendency = profile.carries / total_actions
        goal_threat_tendency = profile.goal_threat / total_actions
        reception_tendency = profile.receptions / total_actions

        # Influence: team-action normalization
        def _safe_ratio(num: int, denom: int) -> float:
            return float(num) / float(denom) if denom > 0 else 0.0

        pass_influence = _safe_ratio(profile.passes, profile.team_passes)
        carry_influence = _safe_ratio(profile.carries, profile.team_carries)
        goal_threat_influence = _safe_ratio(profile.goal_threat, profile.team_goal_threat)
        reception_influence = _safe_ratio(profile.receptions, profile.team_receptions)

        records.append(
            {
                "player_id": pid,
                "spatial_tensor": profile.spatial_tensor,
                # Raw counts
                "passes": profile.passes,
                "carries": profile.carries,
                "goal_threat": profile.goal_threat,
                "receptions": profile.receptions,
                "total_actions": profile.total_actions,
                "team_passes": profile.team_passes,
                "team_carries": profile.team_carries,
                "team_goal_threat": profile.team_goal_threat,
                "team_receptions": profile.team_receptions,
                "team_total_actions": profile.team_total_actions,
                # Tendency features
                "pass_tendency": pass_tendency,
                "carry_tendency": carry_tendency,
                "goal_threat_tendency": goal_threat_tendency,
                "reception_tendency": reception_tendency,
                # Influence features
                "pass_influence": pass_influence,
                "carry_influence": carry_influence,
                "goal_threat_influence": goal_threat_influence,
                "reception_influence": reception_influence,
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    input_dir, output_dir = load_paths()

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in INPUT directory: {input_dir}")

    print(f"Found {len(parquet_files)} final match files in {input_dir}")

    # Step 1: infer pitch bounds from the first file
    sample_df = pd.read_parquet(parquet_files[0])
    (x_min, x_max), (y_min, y_max) = infer_pitch_bounds(sample_df)
    print(f"Inferred pitch bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

    n_bins_x = 50
    n_bins_y = 50
    x_edges = np.linspace(x_min, x_max, n_bins_x + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1, dtype=np.float32)

    players: Dict[int, PlayerSeasonProfile] = {}

    # Step 2–3: iterate over matches and update per-player season profiles
    for idx, path in enumerate(parquet_files, start=1):
        print(f"[{idx}/{len(parquet_files)}] Processing match file: {path.name}")
        df = pd.read_parquet(path)
        update_histograms_for_match(df, players, x_edges, y_edges)

    # Step 4: compute scalar features
    profiles_df = compute_scalar_features(players)

    # Step 5: save final player profiles (one row per player)
    output_path = output_dir / "processed_player_profiles.pkl"
    profiles_df.to_pickle(output_path)
    print(f"\nSaved processed player profiles to: {output_path}")
    print(f"Total players: {len(profiles_df)}")


if __name__ == "__main__":
    main()

