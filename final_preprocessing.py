import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"


def load_paths() -> Tuple[Path, Path]:
    """
    Load input/output directories from creds/gdrive_folder.json.
    """
    if not CREDS_FILE.exists():
        raise FileNotFoundError(f"Missing creds file: {CREDS_FILE}")

    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)

    input_dir = Path(cfg["merged_parquets_folder_path"])
    output_dir = Path(cfg["final_data"])

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def explore_sample_file(input_dir: Path, n_rows: int = 5) -> None:
    """
    Basic exploration step on a single sample parquet file.

    This is meant for quick schema validation before running the full pipeline.
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in input directory: {input_dir}")

    sample_path = parquet_files[0]
    print(f"🔍 Exploring sample file: {sample_path}")

    df_sample = pd.read_parquet(sample_path)
    print("\nColumns:")
    print(df_sample.columns.tolist())
    print("\nDtypes:")
    print(df_sample.dtypes)
    print(f"\nHead ({n_rows} rows):")
    print(df_sample.head(n_rows))


def _extract_match_id_from_filename(path: Path) -> Optional[int]:
    """
    Try to infer a numeric match_id from filenames like 'match_2006551.parquet'.
    """
    m = re.search(r"(\d+)", path.stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _compute_time_seconds(df: pd.DataFrame) -> pd.Series:
    """
    Compute a continuous time axis in seconds for duration calculations.
    Priority:
      1) numeric 'timestamp'
      2) 'minute' + 'second'
      3) 'minute' + 'seconds'
      4) 'second' or 'seconds'
      5) fallback to row index.
    """
    if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
        return df["timestamp"].astype(float)

    if "minute" in df.columns and "second" in df.columns:
        return (df["minute"] * 60 + df["second"]).astype(float)

    if "minute" in df.columns and "seconds" in df.columns:
        return (df["minute"] * 60 + df["seconds"]).astype(float)

    if "second" in df.columns:
        return df["second"].astype(float)

    if "seconds" in df.columns:
        return df["seconds"].astype(float)

    # Final fallback: sequential index as time surrogate
    return pd.Series(range(len(df)), index=df.index, dtype="float64")


def _get_action_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask indicating rows that count as controlled actions.

    Preference order:
      1) 'type' (generic action column)
      2) 'event_type' (StatsBomb-style merged events)
      3) non-null 'event_id'
    """
    if "type" in df.columns:
        return df["type"].notna()
    if "event_type" in df.columns:
        return df["event_type"].notna()
    if "event_id" in df.columns:
        return df["event_id"].notna()
    # If no explicit event/action indicator is available, treat all rows as non-actions
    return pd.Series(False, index=df.index)


def process_match_df(df: pd.DataFrame, match_id_from_filename: Optional[int] = None) -> pd.DataFrame:
    """
    Apply possession-chain segmentation and junk filtering to a single match dataframe.
    """
    if "possession" not in df.columns:
        raise ValueError("Input dataframe must contain a 'possession' column.")

    df = df.copy()

    # Ensure a 'match_id' column is available for grouping if present or inferred
    if "match_id" not in df.columns and match_id_from_filename is not None:
        df["match_id"] = match_id_from_filename

    # Sorting: use period and frame_number if available, otherwise timestamp,
    # then fall back to minute/second combinations.
    sort_cols = []

    if "period" in df.columns:
        sort_cols.append("period")

    if "frame_number" in df.columns:
        sort_cols.append("frame_number")
    elif "timestamp" in df.columns:
        sort_cols.append("timestamp")
    else:
        # Use time components if available
        if "minute" in df.columns:
            sort_cols.append("minute")
        if "second" in df.columns:
            sort_cols.append("second")
        elif "seconds" in df.columns:
            sort_cols.append("seconds")

    if sort_cols:
        df = df.sort_values(sort_cols)

    # Continuous time axis for duration calculations
    df["_time_seconds"] = _compute_time_seconds(df)

    # Controlled actions
    df["_is_action"] = _get_action_mask(df).astype("int64")

    # Grouping keys: match_id (if present) + possession
    group_cols = []
    if "match_id" in df.columns:
        group_cols.append("match_id")
    group_cols.append("possession")

    grouped = df.groupby(group_cols, sort=False)

    # Per-possession stats
    stats = grouped["_time_seconds"].agg(["min", "max"]).rename(columns={"min": "t_min", "max": "t_max"})
    stats["sequence_duration"] = stats["t_max"] - stats["t_min"]
    stats["action_count"] = grouped["_is_action"].sum()

    # Filtering logic:
    # Keep possession if (action_count >= 2) OR (sequence_duration >= 4 seconds)
    stats["keep"] = (stats["action_count"] >= 2) | (stats["sequence_duration"] >= 4.0)

    # Merge keep-flag back to original rows and filter
    stats_reset = stats.reset_index()[group_cols + ["keep"]]
    df = df.merge(stats_reset, on=group_cols, how="left")

    # Some rows may not belong to a valid possession group (e.g., NaN possession),
    # which will give keep = NaN after the merge. Treat those as junk (keep=False).
    keep_mask = df["keep"].fillna(False).astype(bool)
    filtered_df = df[keep_mask].drop(columns=["_time_seconds", "_is_action", "keep"])

    return filtered_df


def process_all_matches(input_dir: Path, output_dir: Path, limit: Optional[int] = None) -> None:
    """
    Iterate over all match parquet files, apply filtering, and save per-match outputs.
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in input directory: {input_dir}")

    if limit is not None:
        parquet_files = parquet_files[:limit]

    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    for idx, path in enumerate(parquet_files, start=1):
        print(f"\n[{idx}/{len(parquet_files)}] Processing match file: {path.name}")

        df = pd.read_parquet(path)

        inferred_match_id = _extract_match_id_from_filename(path)
        if inferred_match_id is not None:
            print(f"  Inferred match_id from filename: {inferred_match_id}")

        filtered_df = process_match_df(df, match_id_from_filename=inferred_match_id)

        # Construct output filename: final_<original_stem>.parquet
        out_name = f"final_{path.stem}.parquet"
        out_path = output_dir / out_name

        print(
            f"  Keeping {len(filtered_df)} rows out of {len(df)} "
            f"({filtered_df['possession'].nunique()} possessions after filtering)"
        )
        print(f"  Saving to: {out_path}")

        filtered_df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final ML preprocessing for football tracking+event data: "
            "segment into possession chains, apply junk filtering, and write per-match parquet files."
        )
    )
    parser.add_argument(
        "--explore-sample",
        action="store_true",
        help="Explore a sample parquet from the merged_parquets_folder_path and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of input parquet files to process.",
    )

    args = parser.parse_args()

    input_dir, output_dir = load_paths()

    if args.explore_sample:
        explore_sample_file(input_dir)
        return

    process_all_matches(input_dir, output_dir, limit=args.limit)


if __name__ == "__main__":
    main()

