import pandas as pd
import numpy as np
import glob
import os
import sys
import json
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import ast
import difflib
import shutil

try:
    import ijson  # streaming JSON parser for large tracking files
    from ijson.common import IncompleteJSONError
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    IncompleteJSONError = Exception  # fallback type
    print("Warning: ijson not available, large tracking JSON parsing may be slow or memory-heavy")

# Import for coordinate transformations
try:
    from mplsoccer import Standardizer
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False
    print("Warning: mplsoccer not available, coordinate transformations will use proportional scaling")

# Try to import tqdm, fallback to basic progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available, using basic progress reporting")

# --------------------------------------------------------------------------------------
# Project paths (loaded from creds/gdrive_folder.json)
# IMPORTANT: this script must NEVER write anything into the StatsBomb (Oakland Roots) dir.
# All outputs (mapping + per-match merged parquets) are written under data_folder_path.
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"

if not CREDS_FILE.exists():
    raise FileNotFoundError(f"Missing creds file: {CREDS_FILE}")

with open(CREDS_FILE, "r") as f:
    _creds = json.load(f)

DATA_FOLDER_PATH = Path(_creds["data_folder_path"])
STATSBOMB_DATA_FOLDER_PATH = Path(_creds["statsbomb_data_folder_path"])

# SkillCorner inputs
SKILLCORNER_DIR = DATA_FOLDER_PATH / "skillcorner"
# For full-season processing, this would be SKILLCORNER_DIR / "tracking".
# For your current workflow, we point to the locally downloaded batch folder.
SKILLCORNER_TRACKING_DIR = Path("/Users/estebansanchez/Desktop/batch10")
SKILLCORNER_MATCHES_CSV = SKILLCORNER_DIR / "matches_df.csv"
SKILLCORNER_PLAYERS_CSV = SKILLCORNER_DIR / "players_df.csv"

# StatsBomb inputs (read-only)
STATSBOMB_EVENTS_FILE = STATSBOMB_DATA_FOLDER_PATH / "USLChampionship_2025.parquet"
STATSBOMB_MATCHES_FILE = STATSBOMB_DATA_FOLDER_PATH / "USLChampionship_2025_matches.parquet"
STATSBOMB_PLAYER_SEASON_FILE = STATSBOMB_DATA_FOLDER_PATH / "USLChampionship_2025_player_season.parquet"

# Outputs (all under thesis data folder; never under StatsBomb folder)
OUTPUT_ROOT_DIR = DATA_FOLDER_PATH / "merged"
OUTPUT_MATCH_DIR = OUTPUT_ROOT_DIR / "individual_matches"
MAPPING_DIR = DATA_FOLDER_PATH / "mappings"
MATCH_ID_MAPPING_FILE = MAPPING_DIR / "skillcorner_statsbomb_match_id_mapping.csv"

# Local staging dir for batching tracking JSONs off Google Drive
LOCAL_TRACKING_STAGING_DIR = Path.home() / "Desktop" / "Data_JSONs"

# Backward-compatible names used in the original script
RAW_DATA_DIR = STATSBOMB_DATA_FOLDER_PATH
PROCESSED_DATA_DIR = DATA_FOLDER_PATH / "processed"  # not used unless you add tracking parquet outputs later
USL_TRACKING_DIR = SKILLCORNER_TRACKING_DIR
USL_DATA_DIR = SKILLCORNER_DIR

# Guardrails: ensure outputs are not pointed at the StatsBomb folder
if str(OUTPUT_ROOT_DIR).startswith(str(STATSBOMB_DATA_FOLDER_PATH)):
    raise ValueError("Output directory resolves inside StatsBomb folder; refusing to run.")

MAPPING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MATCH_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_TRACKING_STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Dictionary mapping role_id to role name
ROLE_MAPPING = {
    0: "Goalkeeper",
    2: "Center Back",
    3: "Left Center Back",
    4: "Right Center Back",
    5: "Left Wing Back",
    6: "Right Wing Back",
    7: "Defensive Midfield",
    9: "Left Midfield",
    10: "Right Midfield",
    11: "Attacking Midfield",
    12: "Left Winger",
    13: "Right Winger",
    14: "Left Forward",
    15: "Center Forward",
    16: "Right Forward",
    19: "Left Back",
    20: "Right Back",
    21: "Left Defensive Midfield",
    22: "Right Defensive Midfield"
}

def categorize_role(role_name):
    """
    Categorize player role into defensive line, midfield line, or attacking line
    
    Args:
        role_name (str): The role name from ROLE_MAPPING
        
    Returns:
        str: Category of the role ('Defender', 'Midfielder', 'Striker', 'Goalkeeper', or 'Unknown')
    """
    if pd.isna(role_name) or role_name is None:
        return 'Unknown'
    
    defenders = [
        'Left Center Back', 'Right Center Back', 'Center Back',
        'Left Back', 'Right Back'
    ]
    
    midfielders = [
        'Left Midfield', 'Right Midfield', 'Attacking Midfield', 
        'Right Defensive Midfield', 'Left Defensive Midfield', 'Defensive Midfield',
        'Left Wing Back', 'Right Wing Back'
    ]
    
    strikers = [
        'Right Forward', 'Left Winger', 'Right Winger',
        'Center Forward', 'Left Forward'
    ]
    
    if role_name == 'Goalkeeper':
        return 'Goalkeeper'
    elif role_name in defenders:
        return 'Defender'
    elif role_name in midfielders:
        return 'Midfielder'
    elif role_name in strikers:
        return 'Striker'
    else:
        return 'Unknown'

def _parse_skillcorner_team_cell(team_cell):
    """
    SkillCorner matches_df.csv stores team objects as strings like:
      \"{'id': 2715, 'short_name': 'Tulsa'}\"
    """
    if pd.isna(team_cell) or team_cell is None:
        return None, None
    if isinstance(team_cell, dict):
        return team_cell.get("id"), team_cell.get("short_name")
    if isinstance(team_cell, str):
        try:
            d = ast.literal_eval(team_cell)
            if isinstance(d, dict):
                return d.get("id"), d.get("short_name")
        except Exception:
            return None, None
    return None, None

def _norm_team_name(name):
    """
    Normalize team names across providers (lightweight, no external deps).
    SkillCorner short_name may be truncated (e.g. 'Charleston Bat').
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).lower().strip()
    # remove common noise tokens
    for tok in [" fc", " sc", " cf", " athletic", " city", " united", " the "]:
        s = s.replace(tok, " ")
    # keep alnum only
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    s = " ".join(s.split())
    return s

def _similarity(a, b):
    """
    Fuzzy similarity in [0, 1] using difflib (stdlib).
    """
    a_n = _norm_team_name(a)
    b_n = _norm_team_name(b)
    if not a_n or not b_n:
        return 0.0
    # if one is a prefix of the other (common with truncated SkillCorner names), boost
    if a_n in b_n or b_n in a_n:
        return 0.95
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()

def get_skillcorner_tracking_files():
    """
    SkillCorner tracking JSON files live in:
      data_folder_path/skillcorner/tracking/
    Filenames look like:
      tracking_usl_championship-2025-2006551.json
    """
    pattern = str(SKILLCORNER_TRACKING_DIR / "tracking_usl_championship-2025-*.json")
    return glob.glob(pattern)

def _extract_skillcorner_match_id_from_tracking_path(tracking_path):
    """
    Parse match_id from:
      tracking_usl_championship-2025-<match_id>.json
    """
    base = os.path.basename(tracking_path)
    # split on '-' then strip .json
    try:
        match_id_str = base.split("-")[-1].split(".")[0]
        return int(match_id_str)
    except Exception:
        return None

def get_skillcorner_tracking_match_ids():
    files = get_skillcorner_tracking_files()
    ids = []
    for fp in files:
        mid = _extract_skillcorner_match_id_from_tracking_path(fp)
        if mid is not None:
            ids.append(mid)
    return set(ids)

def load_skillcorner_matches_df():
    if not SKILLCORNER_MATCHES_CSV.exists():
        raise FileNotFoundError(f"Missing SkillCorner matches file: {SKILLCORNER_MATCHES_CSV}")
    df = pd.read_csv(SKILLCORNER_MATCHES_CSV)
    # parse teams
    home_parsed = df["home_team"].apply(_parse_skillcorner_team_cell)
    away_parsed = df["away_team"].apply(_parse_skillcorner_team_cell)
    df["skillcorner_home_team_id"] = home_parsed.apply(lambda x: x[0])
    df["skillcorner_home_team_name"] = home_parsed.apply(lambda x: x[1])
    df["skillcorner_away_team_id"] = away_parsed.apply(lambda x: x[0])
    df["skillcorner_away_team_name"] = away_parsed.apply(lambda x: x[1])
    # match date
    df["date"] = pd.to_datetime(df["date_time"], errors="coerce").dt.date
    df = df.rename(columns={"id": "skillcorner_match_id"})
    return df

def load_statsbomb_matches_df():
    if not STATSBOMB_MATCHES_FILE.exists():
        raise FileNotFoundError(f"Missing StatsBomb matches file: {STATSBOMB_MATCHES_FILE}")
    sb = pd.read_parquet(STATSBOMB_MATCHES_FILE)
    sb["date"] = pd.to_datetime(sb["match_date"], errors="coerce").dt.date
    return sb

def build_match_id_mapping(date_tolerance_days=1, min_combined_score=1.40, overwrite=False):
    """
    Build (or load) SkillCorner↔StatsBomb match-id mapping.\n\n
    Strategy:\n
    - Use only SkillCorner matches that have a tracking JSON file.\n
    - For each SC match, filter SB matches within ±date_tolerance_days.\n
    - Score candidates by fuzzy similarity(home)+fuzzy similarity(away), consider swapped home/away.\n
    - Pick best candidate above min_combined_score.\n
    """
    if MATCH_ID_MAPPING_FILE.exists() and not overwrite:
        return pd.read_csv(MATCH_ID_MAPPING_FILE)

    sc_matches = load_skillcorner_matches_df()
    sb_matches = load_statsbomb_matches_df()

    tracking_ids = get_skillcorner_tracking_match_ids()
    sc_matches = sc_matches[sc_matches["skillcorner_match_id"].isin(tracking_ids)].copy()

    mappings = []
    unmapped = []

    # index SB by date for quick candidate retrieval
    sb_by_date = {}
    for d, g in sb_matches.groupby("date"):
        sb_by_date[d] = g

    def _candidate_dates(d):
        if pd.isna(d) or d is None:
            return []
        d = pd.to_datetime(d).date()
        return [d + pd.Timedelta(days=delta).to_pytimedelta() for delta in range(-date_tolerance_days, date_tolerance_days + 1)]

    for _, sc in sc_matches.iterrows():
        sc_id = int(sc["skillcorner_match_id"])
        sc_date = sc["date"]
        sc_home = sc["skillcorner_home_team_name"]
        sc_away = sc["skillcorner_away_team_name"]

        # gather SB candidates across tolerance window
        candidates = []
        if sc_date is not None and not pd.isna(sc_date):
            for delta in range(-date_tolerance_days, date_tolerance_days + 1):
                d = (pd.to_datetime(sc_date) + pd.Timedelta(days=delta)).date()
                if d in sb_by_date:
                    candidates.append(sb_by_date[d])
        if not candidates:
            unmapped.append((sc_id, "no_date_candidates"))
            continue

        cand = pd.concat(candidates, ignore_index=True)

        best = None
        best_score = -1.0
        best_swapped = False

        for _, sb in cand.iterrows():
            sb_home = sb["home_team"]
            sb_away = sb["away_team"]
            score_direct = _similarity(sc_home, sb_home) + _similarity(sc_away, sb_away)
            score_swapped = _similarity(sc_home, sb_away) + _similarity(sc_away, sb_home)

            if score_direct >= score_swapped:
                score = score_direct
                swapped = False
            else:
                score = score_swapped
                swapped = True

            if score > best_score:
                best_score = score
                best = sb
                best_swapped = swapped

        if best is None or best_score < min_combined_score:
            unmapped.append((sc_id, f"low_score:{best_score:.3f}"))
            continue

        mappings.append({
            "skillcorner_match_id": sc_id,
            "statsbomb_match_id": int(best["match_id"]),
            "date": str(sc_date),
            "skillcorner_home_team_id": sc["skillcorner_home_team_id"],
            "skillcorner_home_team_name": sc_home,
            "skillcorner_away_team_id": sc["skillcorner_away_team_id"],
            "skillcorner_away_team_name": sc_away,
            "statsbomb_home_team_name": best["home_team"] if not best_swapped else best["away_team"],
            "statsbomb_away_team_name": best["away_team"] if not best_swapped else best["home_team"],
            "match_score": float(best_score),
            "swapped_home_away": bool(best_swapped),
        })

    mapping_df = pd.DataFrame(mappings)
    mapping_df = mapping_df.sort_values(["date", "skillcorner_match_id"], ascending=[True, True])

    mapping_df.to_csv(MATCH_ID_MAPPING_FILE, index=False)

    # Also write helper lists so you can filter StatsBomb events quickly
    try:
        included_sb_ids = set(mapping_df["statsbomb_match_id"].astype(int).tolist()) if not mapping_df.empty else set()
        sb_all = sb_matches[["match_id", "match_date", "home_team", "away_team"]].copy()
        sb_all["match_id"] = sb_all["match_id"].astype(int)

        included_fp = MAPPING_DIR / "statsbomb_matches_with_skillcorner_tracking.csv"
        excluded_fp = MAPPING_DIR / "statsbomb_matches_missing_skillcorner_tracking.csv"

        sb_all[sb_all["match_id"].isin(included_sb_ids)].sort_values("match_date").to_csv(included_fp, index=False)
        sb_all[~sb_all["match_id"].isin(included_sb_ids)].sort_values("match_date").to_csv(excluded_fp, index=False)

        print(f"✅ Wrote helper lists: {included_fp.name}, {excluded_fp.name}")
    except Exception as e:
        print(f"⚠️  Could not write helper SB match lists: {e}")

    if unmapped:
        # Save a small report next to the mapping for manual review
        report_fp = MATCH_ID_MAPPING_FILE.with_suffix(".unmapped.csv")
        pd.DataFrame(unmapped, columns=["skillcorner_match_id", "reason"]).to_csv(report_fp, index=False)

    print(f"✅ Mapping created: {len(mapping_df)} matches -> {MATCH_ID_MAPPING_FILE}")
    if unmapped:
        print(f"⚠️  Unmapped SkillCorner matches with tracking: {len(unmapped)} (see {MATCH_ID_MAPPING_FILE.with_suffix('.unmapped.csv')})")

    return mapping_df

def get_raw_tracking_files():
    """
    Get all raw tracking JSON files
    
    Returns:
        list: List of raw tracking file paths
    """
    tracking_pattern = str(USL_TRACKING_DIR / 'tracking_*.json')
    raw_files = glob.glob(tracking_pattern)
    return raw_files

def get_processed_tracking_files():
    """
    Get all processed tracking parquet files with velocity
    
    Returns:
        set: Set of match IDs that have been processed
    """
    tracking_pattern = str(PROCESSED_DATA_DIR / 'tracking_*_with_velocity.parquet')
    processed_files = glob.glob(tracking_pattern)
    
    processed_match_ids = set()
    for file_path in processed_files:
        # Extract match_id from filename
        filename = os.path.basename(file_path)
        match_id = int(filename.split('_')[1])
        processed_match_ids.add(match_id)
    
    return processed_match_ids

def get_unprocessed_raw_files():
    """
    Get raw tracking files that haven't been processed to parquet with velocity yet
    
    Returns:
        list: List of unprocessed raw tracking file paths
    """
    raw_files = get_raw_tracking_files()
    processed_match_ids = get_processed_tracking_files()
    
    unprocessed_files = []
    for raw_file in raw_files:
        # Extract match_id from filename
        filename = os.path.basename(raw_file)
        match_id = int(filename.split('_')[1].split('.')[0])
        
        if match_id not in processed_match_ids:
            unprocessed_files.append(raw_file)
    
    print(f"Found {len(unprocessed_files)} unprocessed raw tracking files out of {len(raw_files)} total")
    return unprocessed_files

def process_raw_tracking_file(match_id, show_progress=False):
    """
    Process a single raw tracking file to create parquet with velocity
    
    Args:
        match_id (int): Match ID to process
        show_progress (bool): Whether to show detailed progress output
        
    Returns:
        tuple: (match_id, success, error_message)
    """
    try:
        # Path to the process_match.py script
        script_path = PROJECT_ROOT / "src" / "data" / "process_match.py"
        
        cmd = [
            sys.executable, str(script_path), str(match_id),
            '--frame-gap', '25',
            '--output-dir', str(PROCESSED_DATA_DIR)
        ]
        
        if show_progress:
            # Don't capture output so we can see the detailed progress
            print(f"  📋 Loading raw tracking data for match {match_id}...")
            print(f"  🔧 Command: {' '.join(cmd[-3:])}")  # Show relevant parts of command
            print(f"  🔄 Starting detailed processing (this may take several minutes):")
            print("     ↳ Loading JSON data...")
            print("     ↳ Processing tracking frames...")
            print("     ↳ Calculating velocities...")
            print("     ↳ Saving to parquet format...")
            print()
            
            result = subprocess.run(cmd, check=True)
            print(f"\n  ✅ Match {match_id} processing completed successfully!")
        else:
            # Capture output for parallel processing (cleaner logs)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        return match_id, True, None
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Subprocess error: {str(e)}"
        if hasattr(e, 'stderr') and e.stderr:
            error_msg += f" - {e.stderr}"
        return match_id, False, error_msg
    except Exception as e:
        return match_id, False, f"Unexpected error: {str(e)}"

def process_raw_tracking_file_wrapper(args):
    """
    Wrapper function for multiprocessing compatibility
    
    Args:
        args: Tuple containing (match_id, raw_file_path, show_progress)
        
    Returns:
        tuple: (match_id, success, error_message)
    """
    match_id, raw_file_path, show_progress = args
    return process_raw_tracking_file(match_id, show_progress=show_progress)

def process_all_unprocessed_raw_files(max_workers=None, use_parallel=True):
    """
    Process all raw tracking files that haven't been converted to parquet with velocity yet
    
    Args:
        max_workers (int): Maximum number of parallel workers. If None, uses CPU count
        use_parallel (bool): Whether to use parallel processing. If False, processes sequentially
    
    Returns:
        tuple: (total_files, successfully_processed)
    """
    print("🔄 Checking for unprocessed raw tracking files...")
    
    unprocessed_files = get_unprocessed_raw_files()
    
    if not unprocessed_files:
        print("✅ All raw tracking files have already been processed!")
        return 0, 0
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(4, len(unprocessed_files))
    else:
        max_workers = min(max_workers, len(unprocessed_files))
    
    print(f"🔄 Processing {len(unprocessed_files)} raw tracking files...")
    
    if use_parallel and len(unprocessed_files) > 1:
        print(f"🚀 Using parallel processing with {max_workers} workers")
        print("💡 If the process is interrupted by memory, try: --sequential")
        return _process_files_parallel(unprocessed_files, max_workers)
    else:
        print("🔄 Using sequential processing")
        return _process_files_sequential(unprocessed_files)

def _process_files_sequential(unprocessed_files):
    """
    Process files sequentially with detailed progress for each match
    
    Args:
        unprocessed_files (list): List of unprocessed file paths
        
    Returns:
        tuple: (total_files, successfully_processed)
    """
    successfully_processed = 0
    total_processing_time = 0
    
    print("🔄 Sequential processing - showing detailed progress for each match:")
    print("=" * 70)
    
    for i, raw_file in enumerate(unprocessed_files, 1):
        # Extract match_id from filename
        filename = os.path.basename(raw_file)
        match_id = int(filename.split('_')[1].split('.')[0])
        
        print(f"\n📊 [{i}/{len(unprocessed_files)}] Starting Match {match_id}")
        print("-" * 50)
        
        # Show file size information
        file_size_mb = os.path.getsize(raw_file) / (1024 * 1024)
        print(f"📁 File size: {file_size_mb:.1f} MB")
        print(f"⏱️  Estimated time: {file_size_mb/50:.1f}-{file_size_mb/30:.1f} minutes")
        
        # Track processing time
        start_time = time.time()
        
        # Use show_progress=True for detailed output
        match_id_result, success, error_msg = process_raw_tracking_file(match_id, show_progress=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        
        if success:
            print(f"✅ Match {match_id} completed successfully!")
            print(f"⏱️  Actual processing time: {elapsed_minutes:.1f} minutes")
            successfully_processed += 1
            total_processing_time += elapsed_time
        else:
            print(f"❌ Error processing match {match_id}: {error_msg}")
            print(f"⏱️  Time before error: {elapsed_minutes:.1f} minutes")
        
        print("-" * 50)
        
        # Show overall progress and time estimates
        remaining = len(unprocessed_files) - i
        print(f"📈 Progress: {i}/{len(unprocessed_files)} completed, {remaining} remaining")
        
        if i > 0 and successfully_processed > 0:
            avg_time_per_file = total_processing_time / successfully_processed
            estimated_remaining_time = avg_time_per_file * remaining / 60  # in minutes
            print(f"⏱️  Average time per file: {avg_time_per_file/60:.1f} minutes")
            print(f"🕐 Estimated time remaining: {estimated_remaining_time:.1f} minutes ({estimated_remaining_time/60:.1f} hours)")
    
    print(f"\n🎉 Sequential processing complete!")
    print(f"✅ Successfully processed: {successfully_processed}/{len(unprocessed_files)} files")
    
    if successfully_processed > 0:
        total_time_hours = total_processing_time / 3600
        avg_time_minutes = (total_processing_time / successfully_processed) / 60
        print(f"⏱️  Total processing time: {total_time_hours:.1f} hours")
        print(f"📊 Average time per file: {avg_time_minutes:.1f} minutes")
    
    return len(unprocessed_files), successfully_processed

def _process_files_parallel(unprocessed_files, max_workers):
    """
    Process files in parallel using ProcessPoolExecutor
    
    Args:
        unprocessed_files (list): List of unprocessed file paths
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        tuple: (total_files, successfully_processed)
    """
    successfully_processed = 0
    failed_files = []
    
    # Prepare arguments for parallel processing
    processing_args = []
    for raw_file in unprocessed_files:
        filename = os.path.basename(raw_file)
        match_id = int(filename.split('_')[1].split('.')[0])
        processing_args.append((match_id, raw_file, False))  # show_progress=False for parallel
    
    print(f"🚀 Starting parallel processing of {len(processing_args)} files...")
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_match = {
            executor.submit(process_raw_tracking_file_wrapper, args): args[0] 
            for args in processing_args
        }
        
        # Use tqdm for progress bar if available, otherwise basic progress
        if HAS_TQDM:
            with tqdm(total=len(future_to_match), desc="Processing matches", unit="match") as pbar:
                for future in as_completed(future_to_match):
                    match_id = future_to_match[future]
                    try:
                        result_match_id, success, error_msg = future.result()
                        
                        if success:
                            pbar.set_postfix_str(f"✅ Match {result_match_id} completed")
                            successfully_processed += 1
                        else:
                            pbar.set_postfix_str(f"❌ Match {result_match_id} failed")
                            failed_files.append((result_match_id, error_msg))
                            
                    except Exception as e:
                        pbar.set_postfix_str(f"❌ Match {match_id} exception")
                        failed_files.append((match_id, f"Future exception: {str(e)}"))
                    
                    pbar.update(1)
        else:
            # Basic progress without tqdm
            completed = 0
            total_tasks = len(future_to_match)
            for future in as_completed(future_to_match):
                match_id = future_to_match[future]
                completed += 1
                try:
                    result_match_id, success, error_msg = future.result()
                    
                    if success:
                        print(f"  ✅ ({completed}/{total_tasks}) Match {result_match_id} completed")
                        successfully_processed += 1
                    else:
                        print(f"  ❌ ({completed}/{total_tasks}) Match {result_match_id} failed: {error_msg}")
                        failed_files.append((result_match_id, error_msg))
                        
                except Exception as e:
                    print(f"  ❌ ({completed}/{total_tasks}) Match {match_id} exception: {str(e)}")
                    failed_files.append((match_id, f"Future exception: {str(e)}"))
    
    # Print results summary
    print(f"\n✅ Parallel processing complete!")
    print(f"   Successfully processed: {successfully_processed}/{len(unprocessed_files)} files")
    
    if failed_files:
        print(f"   Failed files ({len(failed_files)}):")
        for match_id, error_msg in failed_files:
            print(f"     ❌ Match {match_id}: {error_msg}")
    
    return len(unprocessed_files), successfully_processed

def update_match_id_mappings_automatically():
    """
    Automatically update match ID mappings by finding unmapped matches
    and matching them based on team names and dates (with 1-day tolerance)
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        print("🔄 Automatically updating match ID mappings...")
        
        # Load existing mapping
        if MATCH_ID_MAPPING_FILE.exists():
            existing_mapping = pd.read_csv(MATCH_ID_MAPPING_FILE)
            existing_skillcorner_ids = set(existing_mapping['skillcorner_match_id'])
            print(f"📋 Found {len(existing_mapping)} existing mappings")
        else:
            existing_mapping = pd.DataFrame()
            existing_skillcorner_ids = set()
            print("📋 No existing mapping file found, creating new one")
        
        # Load team mapping
        team_mapping_file = PROJECT_ROOT / 'config' / 'team_id_mapping.csv'
        team_mapping = pd.read_csv(team_mapping_file)
        
        # Create team name mappings
        sc_to_sb_team = dict(zip(team_mapping['skillcorner_team_name'], team_mapping['statsbomb_team_name']))
        sc_to_sb_id = dict(zip(team_mapping['skillcorner_team_name'], team_mapping['statsbomb_team_id']))
        
        # Load SkillCorner matches
        print("📊 Loading SkillCorner matches...")
        sc_matches_file = USL_DATA_DIR / 'all_matches.json'
        if not sc_matches_file.exists():
            print(f"❌ SkillCorner matches file not found: {sc_matches_file}")
            return False
            
        with open(sc_matches_file, 'r') as f:
            sc_matches = json.load(f)
        
        # Load StatsBomb matches
        print("📊 Loading StatsBomb matches...")
        sb_matches_file = RAW_DATA_DIR / 'USLChampionship_2025_matches.parquet'
        if not sb_matches_file.exists():
            print(f"❌ StatsBomb matches file not found: {sb_matches_file}")
            return False
            
        sb_matches = pd.read_parquet(sb_matches_file)
        
        # Convert StatsBomb dates to string format for comparison
        sb_matches['date'] = pd.to_datetime(sb_matches['match_date']).dt.strftime('%Y-%m-%d')
        
        # Find unmapped SkillCorner matches
        unmapped_matches = []
        for match in sc_matches:
            sc_match_id = match['id']
            if sc_match_id not in existing_skillcorner_ids:
                unmapped_matches.append(match)
        
        print(f"🔍 Found {len(unmapped_matches)} unmapped matches")
        
        if not unmapped_matches:
            print("✅ All matches are already mapped!")
            return True
        
        # Find new mappings
        new_mappings = []
        mapped_count = 0
        
        for match in unmapped_matches:
            sc_match_id = match['id']
            sc_date = pd.to_datetime(match['date_time']).strftime('%Y-%m-%d')
            sc_home_team = match['home_team']['short_name']
            sc_away_team = match['away_team']['short_name']
            
            # Check if teams exist in mapping
            if sc_home_team not in sc_to_sb_team or sc_away_team not in sc_to_sb_team:
                continue
            
            sb_home_team = sc_to_sb_team[sc_home_team]
            sb_away_team = sc_to_sb_team[sc_away_team]
            
            # Find matching StatsBomb match with date tolerance
            potential_matches = sb_matches[
                (sb_matches['home_team'] == sb_home_team) &
                (sb_matches['away_team'] == sb_away_team)
            ]
            
            if len(potential_matches) == 0:
                continue
            
            # Check dates with 1-day tolerance
            best_match = None
            min_date_diff = float('inf')
            
            for _, sb_match in potential_matches.iterrows():
                sb_date = sb_match['date']
                date_diff = abs((pd.to_datetime(sc_date) - pd.to_datetime(sb_date)).days)
                
                if date_diff <= 1 and date_diff < min_date_diff:
                    min_date_diff = date_diff
                    best_match = sb_match
            
            if best_match is not None:
                # Get team IDs
                sb_home_id = sc_to_sb_id[sc_home_team]
                sb_away_id = sc_to_sb_id[sc_away_team]
                
                new_mapping = {
                    'skillcorner_match_id': sc_match_id,
                    'statsbomb_match_id': best_match['match_id'],
                    'date': sc_date,
                    'skillcorner_home_team_id': match['home_team']['id'],
                    'skillcorner_home_team_name': sc_home_team,
                    'statsbomb_home_team_id': sb_home_id,
                    'statsbomb_home_team_name': sb_home_team,
                    'skillcorner_away_team_id': match['away_team']['id'],
                    'skillcorner_away_team_name': sc_away_team,
                    'statsbomb_away_team_id': sb_away_id,
                    'statsbomb_away_team_name': sb_away_team
                }
                
                new_mappings.append(new_mapping)
                mapped_count += 1
                
                print(f"   ✅ Mapped match {sc_match_id} → {best_match['match_id']} ({sc_home_team} vs {sc_away_team}) - Date diff: {min_date_diff} days")
        
        if new_mappings:
            print(f"🎯 Found {len(new_mappings)} new mappings")
            
            # Create new mappings DataFrame
            new_mappings_df = pd.DataFrame(new_mappings)
            
            # Combine with existing mappings
            if len(existing_mapping) > 0:
                combined_mapping = pd.concat([existing_mapping, new_mappings_df], ignore_index=True)
            else:
                combined_mapping = new_mappings_df
            
            # Sort by date and match ID
            combined_mapping['date'] = pd.to_datetime(combined_mapping['date'])
            combined_mapping = combined_mapping.sort_values(['date', 'skillcorner_match_id'], ascending=[False, True])
            combined_mapping['date'] = combined_mapping['date'].dt.strftime('%Y-%m-%d')
            
            # Save updated mapping
            combined_mapping.to_csv(MATCH_ID_MAPPING_FILE, index=False)
            print(f"💾 Updated mapping file with {len(combined_mapping)} total mappings")
            
            # Show summary of new mappings
            print(f"\n📊 New mappings summary:")
            for mapping in new_mappings:
                print(f"   Match {mapping['skillcorner_match_id']}: {mapping['skillcorner_home_team_name']} vs {mapping['skillcorner_away_team_name']}")
                print(f"     → StatsBomb {mapping['statsbomb_match_id']}: {mapping['statsbomb_home_team_name']} vs {mapping['statsbomb_away_team_name']}")
                print(f"     Date: {mapping['date']}")
            
            return True
        else:
            print("⚠️  No new mappings found")
            return True
            
    except Exception as e:
        print(f"❌ Error updating match mappings automatically: {e}")
        print("Proceeding with existing mapping file...")
        return False

def update_match_id_mappings():
    """
    Update match ID mappings by running the mapping update process
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        print("🔄 Updating match ID mappings...")
        
        # First try automatic update
        if update_match_id_mappings_automatically():
            print("✅ Automatic match ID mapping update completed!")
            return True
        else:
            # Fallback to manual update
            print("🔄 Falling back to manual update...")
            update_match_mappings()
            print("✅ Manual match ID mappings updated successfully!")
            return True
    except Exception as e:
        print(f"❌ Error updating match mappings: {e}")
        print("Proceeding with existing mapping file...")
        return False

def load_match_mapping():
    """
    Load match ID mapping after ensuring it's updated
    
    Returns:
        pd.DataFrame: Match mapping dataframe
    """
    # Load match ID mapping (don't auto-update here since we do it in main)
    return pd.read_csv(MATCH_ID_MAPPING_FILE)

def load_events_data():
    """
    Load events data from parquet file
    
    Returns:
        pd.DataFrame: Events dataframe
    """
    if not STATSBOMB_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Missing StatsBomb events parquet: {STATSBOMB_EVENTS_FILE}")
    events_df = pd.read_parquet(STATSBOMB_EVENTS_FILE)
    return events_df

def load_pitch_dimensions(skillcorner_match_id):
    """
    Load pitch dimensions from match JSON file
    
    Args:
        skillcorner_match_id (int): SkillCorner match ID
        
    Returns:
        tuple: (pitch_length, pitch_width) or (None, None) if not found
    """
    # Your current SkillCorner metadata CSVs do not include pitch dimensions.\n
    # For now we use standard professional pitch defaults. Coordinate transforms\n
    # from StatsBomb to SkillCorner can still be done proportionally or via mplsoccer.\n
    # If you later add per-match pitch dims, wire them in here.\n
    pitch_length = 105.0
    pitch_width = 68.0
    return pitch_length, pitch_width

def subsample_tracking_data(tracking_df, subsample_seconds=0.2):
    """
    Subsample tracking data to keep frames with timestamps divisible by subsample_seconds.
    Examples:
    - subsample_seconds=0.1: keeps frames at 121.0, 121.1, 121.2, 121.3, etc.
    - subsample_seconds=0.2: keeps frames at 121.0, 121.2, 121.4, 121.6, etc.
    - subsample_seconds=0.5: keeps frames at 121.0, 121.5, 122.0, 122.5, etc.
    
    Args:
        tracking_df (pd.DataFrame): Full tracking data
        subsample_seconds (float): Target interval in seconds (default: 0.2s)
        
    Returns:
        pd.DataFrame: Subsampled tracking data with frames at exact intervals
    """
    print(f"Subsampling tracking data every {subsample_seconds} seconds...")
    
    # Create timestamp if not available
    if 'timestamp' not in tracking_df.columns and 'seconds' in tracking_df.columns:
        tracking_df = tracking_df.copy()
        tracking_df['timestamp'] = tracking_df['minute'] * 60 + tracking_df['seconds']
        print("Created timestamp from minute and seconds columns")
    
    if 'timestamp' not in tracking_df.columns:
        print("Warning: No timestamp column found, cannot perform time-based subsampling")
        return tracking_df
    
    # Group by period and process each period separately
    subsampled_data = []
    
    for period in tracking_df['period'].unique():
        period_data = tracking_df[tracking_df['period'] == period].copy()
        
        # Sort by timestamp
        period_data = period_data.sort_values('timestamp')
        
        # Get unique timestamps
        unique_timestamps = sorted(period_data['timestamp'].unique())
        
        # Select timestamps that are divisible by subsample_seconds
        selected_timestamps = []
        for timestamp in unique_timestamps:
            # Check if timestamp is divisible by subsample_seconds (with small tolerance for floating point)
            remainder = timestamp % subsample_seconds
            if remainder < 0.001 or abs(remainder - subsample_seconds) < 0.001:
                selected_timestamps.append(timestamp)
        
        # Filter data to selected timestamps only
        subsampled_period = period_data[period_data['timestamp'].isin(selected_timestamps)]
        subsampled_data.append(subsampled_period)
        
        # Show examples of selected timestamps
        if selected_timestamps:
            examples = selected_timestamps[:5]
            print(f"  Period {period}: {len(unique_timestamps)} → {len(selected_timestamps)} frames")
            print(f"    Examples: {[f'{t:.1f}s' for t in examples]}")
            
            # Show preservation rate
            preservation_rate = len(selected_timestamps) / len(unique_timestamps) * 100
            print(f"    Preservation rate: {preservation_rate:.1f}%")
        else:
            print(f"  Period {period}: No frames found at {subsample_seconds}s intervals")
    
    # Combine all periods
    result_df = pd.concat(subsampled_data, ignore_index=True)
    
    # Show overall statistics
    original_frames = len(tracking_df['frame_number'].unique())
    result_frames = len(result_df['frame_number'].unique())
    total_preservation = len(result_df) / len(tracking_df) * 100
    
    print(f"Total frames: {original_frames} → {result_frames} ({total_preservation:.1f}% of original data)")
    
    # Show some examples of the selected timestamps
    if len(result_df) > 0:
        all_timestamps = sorted(result_df['timestamp'].unique())
        if len(all_timestamps) >= 5:
            print(f"Sample selected timestamps: {[f'{t:.1f}s' for t in all_timestamps[:5]]}")
    
    return result_df

def filter_priority_events(events_df, priority_types=['Pass', 'Carry', 'Shot']):
    """
    Filter events intelligently: apply priority filtering only when multiple events compete 
    for the same time window, otherwise keep all events
    
    Args:
        events_df (pd.DataFrame): Full events data
        priority_types (list): List of event types to prioritize when there are conflicts
        
    Returns:
        pd.DataFrame: Intelligently filtered events data
    """
    print(f"Applying intelligent event filtering with priority types: {priority_types}")
    
    # Show original event distribution
    original_count = len(events_df)
    event_counts = events_df['type'].value_counts()
    print(f"Original events: {original_count:,}")
    
    # Create timestamp for temporal analysis
    events_df['timestamp'] = (events_df['minute'] * 60 + events_df['second']).astype(float)
    
    # Group events by period and timestamp windows (1-second tolerance)
    # to identify when multiple events compete for the same time window
    events_df['time_window'] = (events_df['timestamp']).round(0)  # Round to nearest second
    
    # Identify time windows with multiple events
    conflicts_by_period = {}
    total_conflicts = 0
    
    for period in events_df['period'].unique():
        period_events = events_df[events_df['period'] == period]
        time_window_counts = period_events.groupby('time_window').size()
        conflict_windows = time_window_counts[time_window_counts > 1].index
        conflicts_by_period[period] = set(conflict_windows)
        total_conflicts += len(conflict_windows)
        
        if len(conflict_windows) > 0:
            print(f"  Period {period}: {len(conflict_windows)} time windows with multiple events")
    
    print(f"Total conflicting time windows: {total_conflicts}")
    
    if total_conflicts == 0:
        print("✅ No event conflicts detected - keeping all events")
        # Remove temporary columns
        result_df = events_df.drop(['time_window'], axis=1)
        return result_df
    
    # Apply intelligent filtering: only filter events in conflicting time windows
    filtered_events = []
    priority_filtered_count = 0
    
    for period in events_df['period'].unique():
        period_events = events_df[events_df['period'] == period].copy()
        conflict_windows = conflicts_by_period[period]
        
        for time_window in period_events['time_window'].unique():
            window_events = period_events[period_events['time_window'] == time_window]
            
            if time_window in conflict_windows:
                # Multiple events in this window - apply priority filtering
                priority_events_in_window = window_events[window_events['type'].isin(priority_types)]
                
                if len(priority_events_in_window) > 0:
                    # Keep priority events only
                    filtered_events.append(priority_events_in_window)
                    priority_filtered_count += len(window_events) - len(priority_events_in_window)
                else:
                    # No priority events in window, keep first event (by timestamp)
                    filtered_events.append(window_events.iloc[:1])
                    priority_filtered_count += len(window_events) - 1
            else:
                # Single event in this window - keep regardless of type
                filtered_events.append(window_events)
    
    # Combine all filtered events
    if filtered_events:
        # Filter out empty DataFrames before concatenation
        valid_events = [df for df in filtered_events if not df.empty and len(df) > 0]
        
        if valid_events:
            try:
                result_df = pd.concat(valid_events, ignore_index=True)
            except Exception as e:
                print(f"⚠️  Error in pd.concat, trying alternative approach: {e}")
                # Fallback: combine manually
                result_df = pd.concat(valid_events, ignore_index=True, sort=False)
        else:
            result_df = events_df.head(0).copy()  # Empty dataframe with same structure
    else:
        result_df = events_df.head(0).copy()  # Empty dataframe with same structure
    
    # Remove temporary columns
    result_df = result_df.drop(['time_window'], axis=1)
    
    # Show filtering results
    filtered_count = len(result_df)
    filtered_event_counts = result_df['type'].value_counts()
    
    print(f"Filtered events: {filtered_count:,} ({filtered_count/original_count*100:.1f}% of original)")
    print(f"Events filtered due to conflicts: {priority_filtered_count:,}")
    print(f"Event distribution after intelligent filtering:")
    for event_type in priority_types:
        if event_type in filtered_event_counts:
            print(f"  {event_type}: {filtered_event_counts[event_type]:,}")
    
    # Show other event types that were preserved
    other_types = filtered_event_counts[~filtered_event_counts.index.isin(priority_types)]
    if len(other_types) > 0:
        print(f"Other preserved event types:")
        for event_type, count in other_types.head(5).items():
            print(f"  {event_type}: {count:,}")
    
    return result_df

def transform_statsbomb_to_skillcorner(x_sb, y_sb, pitch_length, pitch_width):
    """
    Transform coordinates from StatsBomb to SkillCorner using proper standardization.
    
    This function uses mplsoccer's Standardizer to maintain relative positions 
    to pitch markings (penalty areas, center circle, etc.) when converting 
    between data providers, avoiding the problems of naive linear scaling.
    
    Args:
        x_sb (float): StatsBomb x-coordinate (0-120)
        y_sb (float): StatsBomb y-coordinate (0-80)  
        pitch_length (float): Real pitch length in meters (SkillCorner)
        pitch_width (float): Real pitch width in meters (SkillCorner)
        
    Returns:
        tuple: (x_sc, y_sc) in SkillCorner coordinates
    """
    if pd.isna(x_sb) or pd.isna(y_sb) or pd.isna(pitch_length) or pd.isna(pitch_width):
        return None, None
    
    try:
        if HAS_MPLSOCCER:
            # Create standardizer from StatsBomb to SkillCorner
            # StatsBomb uses 120x80 dimensions, SkillCorner uses real pitch dimensions
            standardizer = Standardizer(pitch_from='statsbomb', pitch_to='skillcorner',
                                       length_to=pitch_length, width_to=pitch_width)
            
            # Transform coordinates - standardizer expects arrays
            x_transformed, y_transformed = standardizer.transform(
                np.array([x_sb]), np.array([y_sb])
            )
            
            return float(x_transformed[0]), float(y_transformed[0])
        else:
            # Fallback to proportional scaling if mplsoccer not available
            x_sc = (x_sb / 120.0) * pitch_length
            y_sc = (y_sb / 80.0) * pitch_width
            return x_sc, y_sc
        
    except Exception as e:
        # Fallback to proportional scaling if standardizer fails
        print(f"⚠️ Standardizer failed, using proportional scaling: {e}")
        x_sc = (x_sb / 120.0) * pitch_length
        y_sc = (y_sb / 80.0) * pitch_width
        return x_sc, y_sc

def flip_statsbomb_coordinates(x_sb, y_sb):
    """
    Flip StatsBomb coordinates when team is attacking towards left (attacking_half = 'left').
    
    StatsBomb coordinates are always oriented so teams attack from left to right (0->120).
    But when a team's attacking_half is 'left', they are actually attacking towards x=0,
    so we need to flip the coordinates.
    
    Args:
        x_sb (float): Original StatsBomb x-coordinate (0-120)
        y_sb (float): Original StatsBomb y-coordinate (0-80)
        
    Returns:
        tuple: (flipped_x, flipped_y) in StatsBomb coordinates
    """
    if pd.isna(x_sb) or pd.isna(y_sb):
        return None, None
    
    # Flip x-coordinate: 120 - x (so 0 becomes 120, 120 becomes 0)
    flipped_x = 120.0 - float(x_sb)
    # Flip y-coordinate: 80 - y (so 0 becomes 80, 80 becomes 0)  
    flipped_y = 80.0 - float(y_sb)
    
    return flipped_x, flipped_y

def extract_coordinates_from_location(location):
    """
    Extract x, y coordinates from various location formats.
    
    Args:
        location: Can be list, tuple, numpy array, or string representation
        
    Returns:
        tuple: (x, y) or (None, None) if extraction fails
    """
    if location is None or (hasattr(location, 'size') and location.size == 0):
        return None, None
    
    try:
        # Handle different data types
        if hasattr(location, 'shape'):  # numpy array
            if location.shape[0] >= 2:
                return float(location[0]), float(location[1])
        elif isinstance(location, str):
            # If string, try to parse as list
            import ast
            location = ast.literal_eval(location)
            if len(location) >= 2:
                return float(location[0]), float(location[1])
        elif isinstance(location, (list, tuple)):
            if len(location) >= 2:
                return float(location[0]), float(location[1])
    except (ValueError, TypeError, SyntaxError, IndexError, AttributeError):
        pass
    
    return None, None

def process_tracking_file(tracking_file, events_df, match_mapping, subsample_seconds=0.2, priority_events=['Pass', 'Carry', 'Shot'], preserve_all_frames=False):
    """
    Process a single tracking file and merge it with events data (optimized for memory)
    Uses intelligent event filtering: only applies priority filtering when multiple events 
    compete for the same time window, otherwise preserves all events.
    
    Args:
        tracking_file (str): Path to tracking file
        events_df (pd.DataFrame): Events dataframe
        match_mapping (pd.DataFrame): Match mapping dataframe
        subsample_seconds (float): Interval for subsampling tracking data (default: 0.2s)
        priority_events (list): List of priority event types when resolving conflicts
        
    Returns:
        pd.DataFrame: Merged tracking and events data (memory optimized)
    """
    # Extract match_id from filename
    match_id = int(os.path.basename(tracking_file).split('_')[1])
    
    # Get corresponding StatsBomb match_id
    match_info = match_mapping[match_mapping['skillcorner_match_id'] == match_id]
    if len(match_info) == 0:
        print(f"No mapping found for match {match_id}")
        return None
    
    statsbomb_match_id = match_info['statsbomb_match_id'].iloc[0]
    
    # Load tracking data
    print(f"Loading tracking data from {tracking_file}")
    tracking_df = pd.read_parquet(tracking_file)
    print(f"Original tracking data: {len(tracking_df):,} rows, {len(tracking_df['frame_number'].unique()):,} unique frames")
    
    # Apply frame preservation logic based on requirements
    if preserve_all_frames:
        print("🔧 PRESERVE_ALL_FRAMES mode: Keeping all tracking data for each frame")
    else:
        # Apply intelligent frame preservation to maintain target frequency
        tracking_df = subsample_tracking_data(tracking_df, subsample_seconds=subsample_seconds)
    
    # Add role name based on player_role_id
    tracking_df['role_name'] = tracking_df['player_role_id'].map(ROLE_MAPPING)
    
    # Add line category based on role name
    tracking_df['role_line'] = tracking_df['role_name'].apply(categorize_role)
    
    # Load pitch dimensions
    pitch_length, pitch_width = load_pitch_dimensions(match_id)
    
    # Get events for this match and apply intelligent filtering
    match_events = events_df[events_df['match_id'] == statsbomb_match_id].copy()
    print(f"Processing events for teams: {match_events.team.unique()}")
    
    # Apply intelligent filtering: only filter when there are event conflicts
    match_events = filter_priority_events(match_events, priority_types=priority_events)
    
    # Create timestamp in tracking data
    tracking_df['timestamp'] = (tracking_df['minute'] * 60 + tracking_df['seconds']).astype(float)
    
    # Create timestamp in events data (assuming events have minute and second columns)
    match_events['timestamp'] = (match_events['minute'] * 60 + match_events['second']).astype(float)
    
    # Check for column conflicts and handle them before merge
    conflict_columns = set(tracking_df.columns) & set(match_events.columns)
    print(f"   🔍 Column conflicts detected: {conflict_columns}")
    
    # Rename conflicting columns in events data to avoid loss
    events_rename_map = {}
    for col in conflict_columns:
        if col not in ['timestamp', 'period']:  # Keep merge keys as is
            events_rename_map[col] = f'event_{col}'
    
    if events_rename_map:
        print(f"   🔧 Renaming event columns to avoid conflicts: {events_rename_map}")
        match_events = match_events.rename(columns=events_rename_map)
    
    # Merge tracking data with events
    print(f"   🔍 Pre-merge: tracking_df has player_id: {'player_id' in tracking_df.columns}")
    print(f"   🔍 Pre-merge: events has player_id: {'player_id' in match_events.columns}")
    
    merged_df = pd.merge_asof(
        tracking_df.sort_values('timestamp'),
        match_events.sort_values('timestamp'),
        on='timestamp',
        by='period',
        direction='nearest',
        tolerance=1.0  # 1 second tolerance for matching
    )
    
    print(f"   🔍 Post-merge: merged_df has player_id: {'player_id' in merged_df.columns}")
    
    # Add event information from events data (handle renamed columns)
    merged_df['event_id'] = merged_df.get('id', None)  # Event ID from events data
    
    # Handle event_player - check for renamed version first
    if 'event_player' in merged_df.columns:
        merged_df['event_player'] = merged_df['event_player']
    else:
        merged_df['event_player'] = merged_df.get('player_y', merged_df.get('player', None))
    
    # Handle event_team - check for renamed version first  
    if 'event_team' in merged_df.columns:
        merged_df['event_team'] = merged_df['event_team']
    else:
        merged_df['event_team'] = merged_df.get('team_y', None)
    
    merged_df['event_type'] = merged_df.get('type', None)
    merged_df['event_location'] = merged_df.get('location', None)
    
    # Add new columns for pass and carry analysis
    merged_df['pass_outcome'] = merged_df.get('pass_outcome', None)
    merged_df['event_duration'] = merged_df.get('duration', None)  # Use 'duration' from StatsBomb events
    merged_df['carry_outcome'] = merged_df.get('carry_outcome', None)
    merged_df['pass_recipient'] = merged_df.get('pass_recipient', None)  # Add pass recipient information
    
    # Add team name columns for both SkillCorner and StatsBomb
    # Load team mapping for this match
    team_mapping_file = PROJECT_ROOT / 'config' / 'team_id_mapping.csv'
    team_mapping_df = pd.read_csv(team_mapping_file)
    
    # Get team names for this match
    home_team_info = match_info.iloc[0]
    away_team_info = match_info.iloc[0]  # We'll get the away team from the match data
    
    # Get home and away team IDs from match info
    home_team_id = home_team_info['skillcorner_home_team_id']
    away_team_id = home_team_info['skillcorner_away_team_id']
    
    # Get team names from team mapping
    home_team_skillcorner_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]['skillcorner_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]) > 0 else None
    away_team_skillcorner_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]['skillcorner_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]) > 0 else None
    
    home_team_statsbomb_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]['statsbomb_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]) > 0 else None
    away_team_statsbomb_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]['statsbomb_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]) > 0 else None
    
    # Create team name mapping based on team ID
    team_name_mapping = {
        home_team_id: {
            'skillcorner_name': home_team_skillcorner_name,
            'statsbomb_name': home_team_statsbomb_name
        },
        away_team_id: {
            'skillcorner_name': away_team_skillcorner_name,
            'statsbomb_name': away_team_statsbomb_name
        }
    }
    
    # Add team name columns to merged_df
    merged_df['skillcorner_team_name'] = merged_df['team'].map(lambda x: team_name_mapping.get(x, {}).get('skillcorner_name') if x in team_name_mapping else None)
    merged_df['statsbomb_team_name'] = merged_df['team'].map(lambda x: team_name_mapping.get(x, {}).get('statsbomb_name') if x in team_name_mapping else None)
    
    # Create combined event_end_location from pass_end_location and carry_end_location
    # Initialize with None values
    merged_df['event_end_location'] = None
    
    # For Pass events, use pass_end_location if available
    if 'pass_end_location' in merged_df.columns:
        pass_mask = (merged_df['event_type'] == 'Pass') & (merged_df['pass_end_location'].notna())
        merged_df.loc[pass_mask, 'event_end_location'] = merged_df.loc[pass_mask, 'pass_end_location']
    
    # For Carry events, use carry_end_location if available
    if 'carry_end_location' in merged_df.columns:
        carry_mask = (merged_df['event_type'] == 'Carry') & (merged_df['carry_end_location'].notna())
        merged_df.loc[carry_mask, 'event_end_location'] = merged_df.loc[carry_mask, 'carry_end_location']
    
    # Add match IDs and pitch dimensions
    merged_df['skillcorner_match_id'] = match_id
    merged_df['statsbomb_match_id'] = statsbomb_match_id
    merged_df['pitch_length'] = pitch_length
    merged_df['pitch_width'] = pitch_width
    
    # ===== OPTIMIZED: COORDINATE PROCESSING MOVED TO BATCH SAVE =====
    # Coordinate processing is now done just before saving the batch for better performance
    # This eliminates the slow row-by-row processing during merge
    print(f"   ⚡ Skipping coordinate processing during merge (will be done before save)")
    
    # Initialize empty coordinate columns for compatibility
    coord_columns = [
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    for col in coord_columns:
        merged_df[col] = None
    
    # Preserve ALL tracking columns and add event columns
    # First, get all original tracking columns
    tracking_columns = tracking_df.columns.tolist()
    
    # Define essential event columns to add
    event_columns = [
        'event_id', 'event_player', 'event_team', 'event_type', 'event_location',
        'pass_outcome', 'event_duration', 'carry_outcome', 'event_end_location', 'pass_recipient',
        # New coordinate columns (StatsBomb only - SkillCorner transformation on-demand)
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    # Define additional match info columns
    match_info_columns = ['skillcorner_match_id', 'statsbomb_match_id', 'pitch_length', 'pitch_width', 'skillcorner_team_name', 'statsbomb_team_name']
    
    # Build list of columns to keep, prioritizing tracking columns and avoiding duplicates
    columns_to_keep = []
    
    # Add all tracking columns first (but handle naming conflicts from merge)
    for col in tracking_columns:
        if col in merged_df.columns:
            columns_to_keep.append(col)
        elif col == 'minute' and 'minute_x' in merged_df.columns:
            # Handle minute conflict - after merge, tracking minute becomes minute_x
            columns_to_keep.append('minute_x')
        elif col == 'seconds' and 'seconds' in merged_df.columns:
            # Handle seconds column
            columns_to_keep.append('seconds')
        else:
            # Debug: Track missing columns from tracking data
            if col == 'player_id':
                print(f"⚠️  Warning: {col} from tracking data not found in merged_df.columns")
                print(f"   Available columns: {sorted(merged_df.columns.tolist())}")
            elif col not in ['seconds']:  # seconds gets renamed, so ignore it
                print(f"⚠️  Warning: tracking column '{col}' not found in merged dataframe")
    
    # Add event columns that don't conflict with tracking columns
    for col in event_columns:
        if col in merged_df.columns and col not in columns_to_keep:
            columns_to_keep.append(col)
    
    # Add match info columns
    for col in match_info_columns:
        if col not in columns_to_keep:
            columns_to_keep.append(col)
    
    # Select only available columns from the merged dataframe (remove duplicates)
    available_columns = []
    seen_columns = set()
    for col in columns_to_keep:
        if col in merged_df.columns and col not in seen_columns:
            available_columns.append(col)
            seen_columns.add(col)
    
    print(f"   📋 Selected {len(available_columns)} unique columns (removed {len(columns_to_keep) - len(available_columns)} duplicates)")
    result_df = merged_df[available_columns].copy()
    
    # Rename conflicting columns for clarity (only if they exist and target doesn't exist)
    column_renames = {}
    if 'player_short_name' in result_df.columns and 'player' not in result_df.columns:
        column_renames['player_short_name'] = 'player'
    if 'team_name' in result_df.columns and 'team' not in result_df.columns:
        column_renames['team_name'] = 'team'
    if 'seconds' in result_df.columns and 'second' not in result_df.columns:
        column_renames['seconds'] = 'second'
    if 'minute_x' in result_df.columns and 'minute' not in result_df.columns:
        column_renames['minute_x'] = 'minute'
    
    # Apply renames only if safe to do so
    if column_renames:
        print(f"   🔄 Applying column renames: {column_renames}")
        result_df = result_df.rename(columns=column_renames)
    
    # Final check for duplicate columns
    if len(result_df.columns) != len(set(result_df.columns)):
        duplicates = [col for col in result_df.columns if list(result_df.columns).count(col) > 1]
        print(f"   ⚠️  Found duplicate columns after processing: {list(set(duplicates))}")
        # Keep only the first occurrence of each column
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        print(f"   ✅ Removed duplicates, final columns: {len(result_df.columns)}")
    
    print(f"✅ Preserved {len(result_df.columns)} columns from tracking data ({len(tracking_columns)} original) + events")
    
    return result_df

def check_output_file_columns(output_file):
    """
    Check if the output file has all required columns for behind defense analysis
    
    Args:
        output_file (Path): Path to the output file
        
    Returns:
        tuple: (has_all_columns, missing_columns, needs_reprocessing)
    """
    # Define all columns that should be preserved from tracking data with velocities
    # Note: 'minute' gets renamed to 'minute_x' during merge, then back to 'minute' in final output
    required_tracking_columns = [
        'frame_number', 'minute', 'second', 'period', 'team', 'player_id', 'player_role_id',
        'x', 'y', 'is_detected', 'player_in_possession', 'team_in_possession', 
        'interpolated', 'player_number', 'player', 'team_color',
        'defending_half', 'attacking_half', 'offside', 
        'velocity_x', 'velocity_y', 'velocity_magnitude', 'velocity_capped',
        'role_name', 'role_line'
    ]
    
    # Event-related columns
    required_event_columns = [
        'event_id', 'event_player', 'event_team', 'event_type', 'event_location',
        'pass_outcome', 'event_duration', 'carry_outcome', 'event_end_location', 'pass_recipient',
        # Coordinate columns (StatsBomb only - SkillCorner transformation on-demand)
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    # Match info columns
    required_match_columns = ['skillcorner_match_id', 'statsbomb_match_id', 'pitch_length', 'pitch_width']
    
    # Team name columns
    required_team_columns = ['skillcorner_team_name', 'statsbomb_team_name']
    
    # Combine all required columns
    required_columns = required_tracking_columns + required_event_columns + required_match_columns + required_team_columns
    
    if not output_file.exists():
        print("Output file does not exist - will create new file")
        return False, required_columns, True
    
    try:
        # Read just the first few rows to check columns
        existing_df = pd.read_parquet(output_file)
        # Take only first row to minimize memory usage
        existing_df = existing_df.head(1)
        existing_columns = set(existing_df.columns)
        required_columns_set = set(required_columns)
        
        # Handle special cases where columns might have been renamed during processing
        # Check for alternative column names that are equivalent
        actual_missing = []
        for col in required_columns:
            if col not in existing_columns:
                # Check for known alternatives
                found_alternative = False
                if col == 'minute' and 'minute' in existing_columns:
                    found_alternative = True  # minute should exist as is in final output
                elif col == 'second' and 'seconds' in existing_columns:
                    found_alternative = True  # seconds can be renamed to second
                elif col == 'player' and 'player_short_name' in existing_columns:
                    found_alternative = True  # player_short_name can be renamed to player
                elif col == 'team' and 'team_name' in existing_columns:
                    found_alternative = True  # team_name can be renamed to team
                
                if not found_alternative:
                    actual_missing.append(col)
        
        missing_columns = actual_missing
        has_all_columns = len(missing_columns) == 0
        
        if missing_columns:
            print(f"⚠️  Missing columns in existing file: {sorted(missing_columns)}")
            print("🔄 Need to reprocess all data to include new columns")
            return False, list(missing_columns), True
        else:
            print("✅ Existing file has all required columns")
            return True, [], False
            
    except Exception as e:
        print(f"❌ Error reading existing output file: {e}")
        print("🔄 Will reprocess from scratch")
        return False, required_columns, True

def get_already_processed_matches(output_dir, force_reprocess=False):
    """
    Get list of match IDs that have already been processed
    NEW APPROACH: Reads individual match files instead of consolidated file
    
    Args:
        output_dir (Path): Path to the output directory containing individual match files
        force_reprocess (bool): If True, return empty set to force reprocessing
        
    Returns:
        set: Set of skillcorner_match_ids that have already been processed
    """
    if force_reprocess:
        print("🔄 Force reprocessing enabled - treating all matches as unprocessed")
        return set()
    
    # Look for individual match files
    match_files = []
    try:
        import glob
        # Pattern for individual match files: match_*.parquet
        match_pattern = str(output_dir / 'match_*.parquet')
        match_files = glob.glob(match_pattern)
        
        if match_files:
            print(f"🔍 Found {len(match_files)} individual match files")
            
            # Extract match IDs from filenames
            processed_matches = set()
            for match_file in match_files:
                try:
                    # Extract match_id from filename: match_2006551.parquet -> 2006551
                    filename = os.path.basename(match_file)
                    if filename.startswith('match_') and filename.endswith('.parquet'):
                        match_id = int(filename.replace('match_', '').replace('.parquet', ''))
                        processed_matches.add(match_id)
                except Exception as e:
                    print(f"⚠️  Error parsing filename {filename}: {e}")
                    continue
            
            print(f"✅ Found {len(processed_matches)} processed matches from individual files")
            return processed_matches
        else:
            print("📄 No individual match files found")
            return set()
            
    except Exception as e:
        print(f"❌ Error searching for match files: {e}")
        return set()

def get_unprocessed_tracking_files(tracking_files, processed_matches):
    """
    Filter tracking files to only include those not yet processed
    ULTRA-OPTIMIZED: Fast filtering with detailed reporting
    
    Args:
        tracking_files (list): List of all tracking files
        processed_matches (set): Set of already processed match IDs
        
    Returns:
        list: List of unprocessed tracking files
    """
    unprocessed_files = []
    skipped_files = []
    
    print(f"🔍 Filtering {len(tracking_files)} tracking files against {len(processed_matches)} processed matches...")
    
    for tracking_file in tracking_files:
        try:
            # Extract match_id from filename (more robust parsing)
            filename = os.path.basename(tracking_file)
            
            # Handle different filename patterns
            if filename.startswith('tracking_') and '_with_velocity.parquet' in filename:
                # Pattern: tracking_2006551_with_velocity.parquet
                match_id = int(filename.split('_')[1])
            elif filename.startswith('tracking_') and '.parquet' in filename:
                # Pattern: tracking_2006551.parquet
                match_id = int(filename.split('_')[1].split('.')[0])
            else:
                print(f"⚠️  Unknown filename pattern: {filename}")
                continue
            
            if match_id not in processed_matches:
                unprocessed_files.append(tracking_file)
            else:
                skipped_files.append(match_id)
                
        except (ValueError, IndexError) as e:
            print(f"⚠️  Error parsing filename {tracking_file}: {e}")
            continue
    
    # Detailed reporting
    print(f"📊 File filtering results:")
    print(f"   Total files found: {len(tracking_files)}")
    print(f"   Already processed: {len(skipped_files)}")
    print(f"   To be processed: {len(unprocessed_files)}")
    
    if skipped_files:
        print(f"   Skipped match IDs: {sorted(skipped_files)[:10]}{'...' if len(skipped_files) > 10 else ''}")
    
    if unprocessed_files:
        # Show first few unprocessed files
        unprocessed_ids = []
        for f in unprocessed_files[:5]:
            try:
                match_id = int(os.path.basename(f).split('_')[1])
                unprocessed_ids.append(match_id)
            except:
                pass
        print(f"   Next to process: {unprocessed_ids}")
    
    return unprocessed_files

def process_coordinates_before_save(batch_data, tracking_df_dict):
    """
    Process coordinates in ULTRA-OPTIMIZED vectorized manner just before saving the batch
    MAXIMUM PERFORMANCE with numpy operations and minimal pandas overhead
    
    Args:
        batch_data (pd.DataFrame): Batch of data to process
        tracking_df_dict (dict): Dictionary {match_id: tracking_df} to get attacking_half
        
    Returns:
        pd.DataFrame: Batch with processed coordinates
    """
    print(f"🔄 Processing coordinates for {len(batch_data):,} events in batch...")
    
    # Filter only events with coordinates
    events_mask = batch_data['event_type'].notna()
    events_df = batch_data[events_mask].copy()
    
    if len(events_df) == 0:
        print(f"   ⚠️ No events found for coordinate processing")
        return batch_data
    
    print(f"   📊 Processing coordinates for {len(events_df):,} events...")
    
    # Initialize coordinate columns
    coord_columns = [
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    for col in coord_columns:
        events_df[col] = None
    
    # ULTRA-OPTIMIZED: Extract coordinates using vectorized operations
    print(f"   🔍 Extracting coordinates (ultra-optimized)...")
    
    # Start coordinates - ULTRA-OPTIMIZED: Direct numpy operations
    start_coords = events_df['event_location'].apply(extract_coordinates_from_location)
    events_df['event_location_x_sb'] = start_coords.apply(lambda x: x[0] if x[0] is not None else None)
    events_df['event_location_y_sb'] = start_coords.apply(lambda x: x[1] if x[1] is not None else None)
    
    # End coordinates - ULTRA-OPTIMIZED: Direct numpy operations
    end_coords = events_df['event_end_location'].apply(extract_coordinates_from_location)
    events_df['event_end_location_x_sb'] = end_coords.apply(lambda x: x[0] if x[0] is not None else None)
    events_df['event_end_location_y_sb'] = end_coords.apply(lambda x: x[1] if x[1] is not None else None)
    
    # ULTRA-OPTIMIZED: Get attacking_half using vectorized operations
    print(f"   🎯 Determining attacking directions (ultra-optimized)...")
    
    # Create a more efficient team mapping
    team_attacking_half = {}
    
    # Use groupby for better performance
    for match_id in events_df['skillcorner_match_id'].unique():
        if match_id in tracking_df_dict:
            tracking_df = tracking_df_dict[match_id]
            # Get unique teams for this match more efficiently
            match_teams = events_df[events_df['skillcorner_match_id'] == match_id]['event_team'].dropna().unique()
            
            for team in match_teams:
                team_tracking = tracking_df[tracking_df['team'] == team]
                if not team_tracking.empty:
                    team_attacking_half[team] = team_tracking['attacking_half'].iloc[0]
    
    # ULTRA-OPTIMIZED: Apply flip using pure numpy operations
    print(f"   🔄 Applying coordinate flips (ultra-optimized)...")
    
    # Create a mapping series for attacking_half
    events_df['team_attacking_half'] = events_df['event_team'].map(team_attacking_half)
    
    # Create masks for teams that need flipping
    needs_flip_mask = events_df['team_attacking_half'] == 'left'
    
    # ULTRA-OPTIMIZED: Initialize flipped columns with original values (numpy arrays)
    events_df['event_location_x_sb_flipped'] = events_df['event_location_x_sb'].values
    events_df['event_location_y_sb_flipped'] = events_df['event_location_y_sb'].values
    events_df['event_end_location_x_sb_flipped'] = events_df['event_end_location_x_sb'].values
    events_df['event_end_location_y_sb_flipped'] = events_df['event_end_location_y_sb'].values
    
    # ULTRA-OPTIMIZED: Apply vectorized flip only where needed using pure numpy
    if needs_flip_mask.any():
        # Get numpy arrays directly for maximum speed
        start_x = events_df['event_location_x_sb'].values
        start_y = events_df['event_location_y_sb'].values
        end_x = events_df['event_end_location_x_sb'].values
        end_y = events_df['event_end_location_y_sb'].values
        
        # Create flip mask as numpy array
        flip_mask = needs_flip_mask.values
        
        # ULTRA-OPTIMIZED: Pure numpy operations for maximum speed
        # Vectorized flip: 120 - x, 80 - y (only where flip_mask is True)
        start_x_flipped = np.where(flip_mask & pd.notna(start_x), 120.0 - start_x, start_x)
        start_y_flipped = np.where(flip_mask & pd.notna(start_y), 80.0 - start_y, start_y)
        end_x_flipped = np.where(flip_mask & pd.notna(end_x), 120.0 - end_x, end_x)
        end_y_flipped = np.where(flip_mask & pd.notna(end_y), 80.0 - end_y, end_y)
        
        # ULTRA-OPTIMIZED: Direct assignment using numpy arrays
        events_df['event_location_x_sb_flipped'] = start_x_flipped
        events_df['event_location_y_sb_flipped'] = start_y_flipped
        events_df['event_end_location_x_sb_flipped'] = end_x_flipped
        events_df['event_end_location_y_sb_flipped'] = end_y_flipped
        
        print(f"   ✅ Applied ultra-optimized flip to {needs_flip_mask.sum():,} events")
    
    # ULTRA-OPTIMIZED: Update original batch using numpy operations
    # First, ensure all columns exist in batch_data
    missing_cols = set(events_df.columns) - set(batch_data.columns)
    if missing_cols:
        print(f"   📋 Adding {len(missing_cols)} missing columns to batch_data")
        for col in missing_cols:
            batch_data[col] = None
    
    # ULTRA-OPTIMIZED: Update only the rows that have events using numpy indexing
    for col in events_df.columns:
        if col in batch_data.columns:
            batch_data.loc[events_mask, col] = events_df[col].values
    
    print(f"   ✅ Ultra-optimized coordinate processing completed!")
    return batch_data

def append_to_output_file(new_data, output_file, tracking_df_dict=None, process_coordinates=True):
    """
    Append new data to existing output file efficiently (memory-safe)
    NOW WITH COORDINATE PROCESSING BEFORE SAVING AND ULTRA LOW MEMORY MODE
    
    Args:
        new_data (pd.DataFrame): New data to append
        output_file (Path): Path to output file
        tracking_df_dict (dict): Dictionary of tracking dataframes for coordinate processing
        process_coordinates (bool): Whether to process coordinates before saving
    """
    if process_coordinates and tracking_df_dict is not None:
        print(f"🔄 Processing coordinates before saving...")
        new_data = process_coordinates_before_save(new_data, tracking_df_dict)
    
    # Check available memory before proceeding
    import psutil
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    print(f"💾 Available memory: {available_gb:.1f} GB")
    
    if output_file.exists():
        print(f"Appending {len(new_data):,} new rows to existing file...")
        
        # FOR ULTRA-LARGE FILES: Use chunked append strategy
        if available_gb < 4.0:  # Less than 4GB available
            print(f"  ⚠️  Low memory detected! Using ultra-safe append mode...")
            return append_with_ultra_low_memory(new_data, output_file)
        
        # Use chunked processing for very large files to avoid memory issues
        try:
            existing_df = pd.read_parquet(output_file)
            existing_count = len(existing_df)
            
            print(f"  Existing rows: {existing_count:,}")
            print(f"  New rows: {len(new_data):,}")
            
            # Estimate memory usage
            total_rows = existing_count + len(new_data)
            estimated_memory_gb = (total_rows * len(new_data.columns) * 8) / (1024**3)  # Rough estimate
            
            if estimated_memory_gb > available_gb * 0.5:  # Use more than 50% of available memory
                print(f"  ⚠️  Large dataset detected! Estimated memory: {estimated_memory_gb:.1f}GB")
                print(f"  🔄 Using chunked processing...")
                return append_with_chunked_processing(new_data, output_file, existing_df)
            
            # Standard processing for smaller datasets
            return append_standard_processing(new_data, output_file, existing_df)
            
        except (MemoryError, Exception) as e:
            print(f"  💥 Error during standard processing: {str(e)}")
            print(f"  🔄 Falling back to ultra-safe mode...")
            return append_with_ultra_low_memory(new_data, output_file)
            
    else:
        combined_df = new_data
        print(f"Creating new output file with {len(new_data):,} rows")
    
    # Save with maximum compression for efficiency
    try:
        combined_df.to_parquet(
            output_file, 
            index=False,
            compression='snappy',  # Fast compression
            engine='pyarrow'       # Most efficient engine
        )
        
        # Clear memory
        del combined_df
        
        print(f"✅ Data saved successfully to {output_file}")
        return None  # Don't return large dataframe to save memory
        
    except Exception as e:
        print(f"💥 Error saving to parquet: {str(e)}")
        print(f"🔄 Trying with different compression...")
        
        # Try with different settings for ultra-large files
        combined_df.to_parquet(
            output_file, 
            index=False,
            compression='gzip',    # Higher compression ratio
            engine='pyarrow',
            row_group_size=50000   # Smaller row groups for memory efficiency
        )
        
        del combined_df
        print(f"✅ Data saved successfully with fallback settings to {output_file}")
        return None

def save_match_data_separately(new_data, output_dir, tracking_df_dict=None, process_coordinates=True):
    """
    Save match data to individual files (one per match_id)
    NEW APPROACH: Much more efficient than appending to a single large file
    
    Args:
        new_data (DataFrame): New data to save
        output_dir (Path): Directory to save individual match files
        tracking_df_dict (dict): Dictionary of tracking dataframes for coordinate processing
        process_coordinates (bool): Whether to process coordinates before saving
    """
    if new_data.empty:
        print("⚠️  No new data to save")
        return
    
    print(f"💾 Saving {len(new_data):,} rows to individual match files...")
    
    # Process coordinates if requested
    if process_coordinates and tracking_df_dict is not None:
        print("🔄 Processing coordinates before saving...")
        new_data = process_coordinates_before_save(new_data, tracking_df_dict)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group data by match_id and save each group separately
    match_groups = new_data.groupby('skillcorner_match_id')
    total_matches = len(match_groups)
    
    print(f"📊 Saving {total_matches} matches to individual files...")
    
    saved_matches = []
    for match_id, match_data in match_groups:
        try:
            # Create filename for this match
            match_filename = f"match_{match_id}.parquet"
            match_filepath = output_dir / match_filename
            
            # Save match data
            match_data.to_parquet(match_filepath, index=False, compression='snappy')
            
            saved_matches.append(match_id)
            print(f"   ✅ Saved match {match_id}: {len(match_data):,} rows -> {match_filename}")
            
        except Exception as e:
            print(f"   ❌ Error saving match {match_id}: {e}")
            continue
    
    print(f"📊 Successfully saved {len(saved_matches)} matches")
    return saved_matches

def append_with_ultra_low_memory(new_data, output_file):
    """
    Ultra-safe append method for very low memory systems
    Saves new data to a separate file to avoid memory issues
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    temp_output = output_file.with_suffix(f'.append_{timestamp}.parquet')
    
    try:
        new_data.to_parquet(
            temp_output,
            index=False,
            compression='gzip',  # Better compression for storage
            engine='pyarrow',
            row_group_size=25000  # Small row groups for memory efficiency
        )
        
        print(f"  💾 New data saved to temporary file: {temp_output}")
        print(f"  ⚠️  Please manually combine files when system has sufficient memory:")
        print(f"     1. Original file: {output_file}")
        print(f"     2. New data file: {temp_output}")
        print(f"  💡 Suggested merge command:")
        print(f"     python -c \"import pandas as pd; pd.concat([pd.read_parquet('{output_file}'), pd.read_parquet('{temp_output}')]).to_parquet('{output_file}', index=False)\"")
        
        return None
        
    except Exception as e:
        print(f"💥 Critical error even in ultra-safe mode: {str(e)}")
        print(f"  Saving as CSV instead...")
        csv_output = temp_output.with_suffix('.csv')
        new_data.to_csv(csv_output, index=False)
        print(f"  📄 Data saved as CSV: {csv_output}")
        return None

def append_with_chunked_processing(new_data, output_file, existing_df):
    """
    Process large datasets using chunked approach to manage memory
    """
    try:
        # Process column alignment first
        existing_cols = set(existing_df.columns)
        new_cols = set(new_data.columns)
        
        if existing_cols != new_cols:
            print(f"  🔧 Aligning columns...")
            missing_in_new = existing_cols - new_cols
            missing_in_existing = new_cols - existing_cols
            
            for col in missing_in_new:
                new_data[col] = None
            for col in missing_in_existing:
                existing_df[col] = None
                
            # Reorder columns
            all_columns = list(new_data.columns)
            existing_df = existing_df[all_columns]
        
        # Create temporary backup
        backup_file = output_file.with_suffix('.backup.parquet')
        print(f"  💾 Creating backup: {backup_file.name}")
        existing_df.to_parquet(backup_file, index=False, compression='snappy')
        
        # Clear existing_df from memory
        del existing_df
        
        # Append new data directly to original file
        print(f"  🔄 Writing combined data in chunks...")
        
        # Read backup and new data in chunks
        import pyarrow.parquet as pq
        
        # Write to temporary file first
        temp_output = output_file.with_suffix('.temp.parquet')
        
        # Combine data using pyarrow for better memory efficiency
        backup_table = pq.read_table(backup_file)
        new_table = pq.Table.from_pandas(new_data)
        
        # Concatenate tables (more memory efficient than pandas)
        combined_table = pq.concat_tables([backup_table, new_table])
        
        # Write with optimized settings
        pq.write_table(
            combined_table, 
            temp_output,
            compression='snappy',
            row_group_size=100000  # Optimize for memory
        )
        
        # Replace original file
        import shutil
        shutil.move(str(temp_output), str(output_file))
        
        # Clean up
        backup_file.unlink()
        
        print(f"  ✅ Chunked processing completed successfully!")
        return None
        
    except Exception as e:
        print(f"  💥 Error in chunked processing: {str(e)}")
        print(f"  🔄 Falling back to ultra-safe mode...")
        return append_with_ultra_low_memory(new_data, output_file)

def append_standard_processing(new_data, output_file, existing_df):
    """
    Standard processing for datasets that fit comfortably in memory
    """
    try:
        # Check for column compatibility
        existing_cols = set(existing_df.columns)
        new_cols = set(new_data.columns)
        
        if existing_cols != new_cols:
            print(f"  ⚠️  Column mismatch detected:")
            print(f"     Existing columns: {len(existing_cols)}")
            print(f"     New columns: {len(new_cols)}")
            
            # Find differences
            missing_in_new = existing_cols - new_cols
            missing_in_existing = new_cols - existing_cols
            
            if missing_in_new:
                print(f"     Missing in new data: {sorted(list(missing_in_new)[:5])}{'...' if len(missing_in_new) > 5 else ''}")
                for col in missing_in_new:
                    new_data[col] = None
            
            if missing_in_existing:
                print(f"     Missing in existing data: {sorted(list(missing_in_existing)[:5])}{'...' if len(missing_in_existing) > 5 else ''}")
                for col in missing_in_existing:
                    existing_df[col] = None
            
            # Reorder columns to match
            all_columns = list(new_data.columns)
            existing_df = existing_df[all_columns]
            
            print(f"  ✅ Column alignment completed")
        
        # Combine with new data
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        print(f"  Combined successfully: {len(combined_df):,} total rows")
        
        # Save with optimized settings
        combined_df.to_parquet(
            output_file, 
            index=False,
            compression='snappy',
            engine='pyarrow'
        )
        
        # Clear memory
        del combined_df
        del existing_df
        
        print(f"  ✅ Standard processing completed successfully!")
        return None
        
    except MemoryError as e:
        print(f"  💥 Memory error in standard processing: {str(e)}")
        del existing_df  # Clean up
        return append_with_ultra_low_memory(new_data, output_file)

def analyze_processing_status(output_dir, tracking_files):
    """
    Analyze the current processing status and provide detailed information
    about what has been processed and what remains to be done
    NEW APPROACH: Works with individual match files instead of consolidated file
    
    Args:
        output_dir (Path): Path to the output directory containing individual match files
        tracking_files (list): List of all tracking files
    """
    print("\n📊 PROCESSING STATUS ANALYSIS")
    print("=" * 50)
    
    # Get all processed matches from individual files
    all_processed_matches = get_already_processed_matches(output_dir, force_reprocess=False)
    
    # Check individual match files status
    try:
        import glob
        match_pattern = str(output_dir / 'match_*.parquet')
        match_files = glob.glob(match_pattern)
        
        if match_files:
            print(f"📄 Individual match files status:")
            print(f"   Directory: {output_dir}")
            print(f"   Total match files: {len(match_files)}")
            
            # Calculate total size
            total_size_mb = 0
            total_rows = 0
            
            for match_file in match_files:
                try:
                    file_size_mb = Path(match_file).stat().st_size / (1024**2)
                    total_size_mb += file_size_mb
                    
                    # Count rows in this file
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(match_file)
                    file_rows = parquet_file.metadata.num_rows
                    total_rows += file_rows
                    
                except Exception as e:
                    print(f"   ⚠️  Error reading {os.path.basename(match_file)}: {e}")
            
            print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
            print(f"   Total rows: {total_rows:,}")
            print(f"   Processed matches: {len(all_processed_matches)}")
            
        else:
            print(f"📄 No individual match files found in: {output_dir}")
            total_rows = 0
            
    except Exception as e:
        print(f"❌ Error analyzing match files: {e}")
        total_rows = 0
    
    # Analyze tracking files
    print(f"\n📁 Tracking files analysis:")
    print(f"   Total tracking files: {len(tracking_files)}")
    
    # Count processed vs unprocessed
    unprocessed_count = 0
    processed_count = 0
    
    for tracking_file in tracking_files:
        try:
            filename = os.path.basename(tracking_file)
            if filename.startswith('tracking_') and '_with_velocity.parquet' in filename:
                match_id = int(filename.split('_')[1])
            elif filename.startswith('tracking_') and '.parquet' in filename:
                match_id = int(filename.split('_')[1].split('.')[0])
            else:
                continue
                
            if match_id in all_processed_matches:
                processed_count += 1
            else:
                unprocessed_count += 1
                
        except:
            continue
    
    print(f"   Already processed: {processed_count}")
    print(f"   Remaining to process: {unprocessed_count}")
    
    if unprocessed_count > 0:
        progress_percent = (processed_count / (processed_count + unprocessed_count)) * 100
        print(f"   Progress: {progress_percent:.1f}% complete")
        
        # Estimate remaining processing time
        if processed_count > 0 and total_rows > 0:
            avg_rows_per_match = total_rows / len(all_processed_matches) if all_processed_matches else 0
            if avg_rows_per_match > 0:
                estimated_remaining_rows = unprocessed_count * avg_rows_per_match
                print(f"   Estimated remaining rows: {estimated_remaining_rows:,.0f}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Individual match files: {len(match_files) if 'match_files' in locals() else 0}")
    print(f"   Total unique matches: {len(all_processed_matches)}")
    print(f"   Files to process: {unprocessed_count}")
    
    print("=" * 50)

def _parse_tracking_timestamp_seconds(ts):
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    s = str(ts)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except Exception:
        return None
    return None

def load_skillcorner_players_df():
    if not SKILLCORNER_PLAYERS_CSV.exists():
        raise FileNotFoundError(f"Missing SkillCorner players file: {SKILLCORNER_PLAYERS_CSV}")
    df = pd.read_csv(SKILLCORNER_PLAYERS_CSV)
    # normalize column names used later
    df = df.rename(columns={"match_id": "skillcorner_match_id"})
    return df

def load_tracking_long_df_from_json(tracking_json_path, match_id, players_match_df, subsample_seconds=0.2, preserve_all_frames=False):
    if not HAS_IJSON:
        raise ImportError("ijson is required to stream tracking JSON files. Install it in your venv: pip install ijson")

    rows = []

    # tolerance: timestamp strings are usually 2 decimals; treat near-multiples as multiples
    def _keep_ts(ts_sec):
        if preserve_all_frames:
            return True
        if ts_sec is None:
            return False
        ts_r = round(float(ts_sec), 2)
        step = float(subsample_seconds)
        k = round(ts_r / step)
        return abs(ts_r - k * step) < 1e-6

    # Google Drive mounts can timeout under heavy sequential reads. We retry
    # a few times before giving up on this match (but continue with others).
    max_retries = 3
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            with open(tracking_json_path, "rb") as f:
                for frame in ijson.items(f, "item"):
                    period = frame.get("period")
                    ts = frame.get("timestamp")
                    frame_num = frame.get("frame")

                    if period is None or ts is None:
                        continue

                    ts_sec = _parse_tracking_timestamp_seconds(ts)
                    if not _keep_ts(ts_sec):
                        continue

                    ball = frame.get("ball_data") or {}
                    ball_x = ball.get("x")
                    ball_y = ball.get("y")
                    ball_z = ball.get("z")
                    ball_detected = ball.get("is_detected")

                    minute = int(ts_sec // 60)
                    second = float(ts_sec - minute * 60)

                    for p in (frame.get("player_data") or []):
                        rows.append({
                            "skillcorner_match_id": match_id,
                            "frame_number": frame_num,
                            "period": int(period),
                            "timestamp": float(ts_sec),
                            "minute": minute,
                            "seconds": second,
                            "player_id": p.get("player_id"),
                            "x": p.get("x"),
                            "y": p.get("y"),
                            "is_detected": p.get("is_detected"),
                            "ball_x": ball_x,
                            "ball_y": ball_y,
                            "ball_z": ball_z,
                            "ball_is_detected": ball_detected,
                        })
            # if we got here, read succeeded
            last_err = None
            break
        except IncompleteJSONError as e:
            # File is truncated / incomplete JSON; skip this match but don't crash the run
            print(f"⚠️  Incomplete JSON for {tracking_json_path}: {e}. Skipping this match.")
            return pd.DataFrame()
        except TimeoutError as e:
            last_err = e
            print(f"⚠️  Timeout reading {tracking_json_path} (attempt {attempt}/{max_retries})")
            time.sleep(2.0)
        except OSError as e:
            # macOS network/FS timeouts can surface as OSError with errno 60
            if getattr(e, "errno", None) == 60:
                last_err = e
                print(f"⚠️  OS timeout reading {tracking_json_path} (attempt {attempt}/{max_retries})")
                time.sleep(2.0)
            else:
                raise

    if last_err is not None:
        print(f"❌ Giving up on {tracking_json_path} after repeated timeouts; skipping this match.")
        return pd.DataFrame()

    tracking_df = pd.DataFrame(rows)
    if tracking_df.empty:
        return tracking_df

    # attach team + position from players CSV
    # players_match_df is already filtered to this match
    players_cols = ["player_id", "team_id", "player_name", "number", "position"]
    players_small = players_match_df[players_cols].drop_duplicates(subset=["player_id"])
    tracking_df = tracking_df.merge(players_small, on="player_id", how="left")

    # rename to align with downstream naming
    tracking_df = tracking_df.rename(columns={"team_id": "team", "position": "role_name"})
    tracking_df["role_line"] = tracking_df["role_name"].apply(categorize_role)

    return tracking_df

def _get_already_merged_skillcorner_match_ids(output_dir, force_reprocess=False):
    if force_reprocess:
        return set()
    pattern = str(Path(output_dir) / "match_*.parquet")
    files = glob.glob(pattern)
    ids = set()
    for fp in files:
        base = os.path.basename(fp)
        try:
            ids.add(int(base.split("_")[1].split(".")[0]))
        except Exception:
            continue
    return ids

def process_tracking_json_and_merge(tracking_json_path, events_df, match_mapping, players_df, subsample_seconds=0.2, priority_events=None, preserve_all_frames=False):
    if priority_events is None:
        priority_events = ["Pass", "Carry", "Shot"]

    sc_match_id = _extract_skillcorner_match_id_from_tracking_path(tracking_json_path)
    if sc_match_id is None:
        print(f"⚠️  Could not parse match id from: {tracking_json_path}")
        return None, None

    mm = match_mapping[match_mapping["skillcorner_match_id"] == sc_match_id]
    if mm.empty:
        print(f"⚠️  No StatsBomb mapping for SkillCorner match {sc_match_id}")
        return None, sc_match_id

    sb_match_id = int(mm["statsbomb_match_id"].iloc[0])

    players_match_df = players_df[players_df["skillcorner_match_id"] == sc_match_id]
    tracking_df = load_tracking_long_df_from_json(
        tracking_json_path,
        match_id=sc_match_id,
        players_match_df=players_match_df,
        subsample_seconds=subsample_seconds,
        preserve_all_frames=preserve_all_frames,
    )
    if tracking_df.empty:
        print(f"⚠️  Empty tracking dataframe for match {sc_match_id}")
        return None, sc_match_id

    pitch_length, pitch_width = load_pitch_dimensions(sc_match_id)

    match_events = events_df[events_df["match_id"] == sb_match_id].copy()
    if match_events.empty:
        print(f"⚠️  No events found for StatsBomb match {sb_match_id} (SC {sc_match_id})")

    match_events = filter_priority_events(match_events, priority_types=priority_events)
    match_events["timestamp"] = (match_events["minute"] * 60 + match_events["second"]).astype(float)

    # Avoid column collisions
    conflict_columns = set(tracking_df.columns) & set(match_events.columns)
    events_rename_map = {c: f"event_{c}" for c in conflict_columns if c not in ["timestamp", "period"]}
    if events_rename_map:
        match_events = match_events.rename(columns=events_rename_map)

    merged_df = pd.merge_asof(
        tracking_df.sort_values("timestamp"),
        match_events.sort_values("timestamp"),
        on="timestamp",
        by="period",
        direction="nearest",
        tolerance=1.0,
    )

    merged_df["event_id"] = merged_df.get("id", None)
    merged_df["event_type"] = merged_df.get("type", None)
    merged_df["event_team"] = merged_df.get("team_y", merged_df.get("team", None))
    merged_df["event_player"] = merged_df.get("player", None)
    merged_df["event_location"] = merged_df.get("location", None)
    merged_df["event_duration"] = merged_df.get("duration", None)
    merged_df["pass_outcome"] = merged_df.get("pass_outcome", None)
    merged_df["carry_outcome"] = merged_df.get("carry_outcome", None)
    merged_df["pass_recipient"] = merged_df.get("pass_recipient", None)

    merged_df["skillcorner_match_id"] = sc_match_id
    merged_df["statsbomb_match_id"] = sb_match_id
    merged_df["pitch_length"] = pitch_length
    merged_df["pitch_width"] = pitch_width

    # note: coordinate transforms can be added later; current focus is correct match-id mapping + per-match files
    return merged_df, sc_match_id

def main(subsample_seconds=0.2, priority_events=None, force_reprocess=False, mapping_overwrite=False, mapping_only=False, preserve_all_frames=False, match_ids=None, batch_size=20):
    print("Starting merge pipeline (SkillCorner tracking JSON + StatsBomb events).")
    print(f"SkillCorner tracking dir: {SKILLCORNER_TRACKING_DIR}")
    print(f"StatsBomb events parquet (read-only): {STATSBOMB_EVENTS_FILE}")
    print(f"StatsBomb matches parquet (read-only): {STATSBOMB_MATCHES_FILE}")
    print(f"Mapping CSV output: {MATCH_ID_MAPPING_FILE}")
    print(f"Per-match output dir: {OUTPUT_MATCH_DIR}")

    match_mapping = build_match_id_mapping(overwrite=mapping_overwrite)
    if mapping_only:
        return

    players_df = load_skillcorner_players_df()
    events_df = load_events_data()

    # All tracking JSONs live on Google Drive; to avoid timeouts when streaming,
    # we process them in small batches via a local staging directory.
    remote_files = sorted(get_skillcorner_tracking_files())
    if not remote_files:
        raise FileNotFoundError(f"No tracking JSON files found in {SKILLCORNER_TRACKING_DIR}")

    if match_ids:
        match_ids_set = set(int(x) for x in match_ids)
        remote_files = [fp for fp in remote_files if _extract_skillcorner_match_id_from_tracking_path(fp) in match_ids_set]
        print(f"Filtering to requested match_ids: {sorted(match_ids_set)} (files={len(remote_files)})")

    # Build work list: (skillcorner_match_id, remote_path)
    work_items = []
    for fp in remote_files:
        sc_id = _extract_skillcorner_match_id_from_tracking_path(fp)
        if sc_id is None:
            continue
        work_items.append((sc_id, fp))

    already_done = _get_already_merged_skillcorner_match_ids(OUTPUT_MATCH_DIR, force_reprocess=force_reprocess)
    print(f"Tracking JSON files found: {len(work_items)}")
    print(f"Already merged matches (skipping): {len(already_done)}")

    # Drop already processed matches
    work_items = [(sc_id, fp) for sc_id, fp in work_items if sc_id not in already_done]
    print(f"Remaining to process: {len(work_items)}")

    processed = 0
    skipped = 0

    if batch_size is None or batch_size <= 0:
        batch_size = 20

    for batch_start in range(0, len(work_items), batch_size):
        batch = work_items[batch_start : batch_start + batch_size]
        if not batch:
            continue

        print(f"\n📦 Processing batch {batch_start // batch_size + 1}: {len(batch)} matches")

        # Clear staging dir
        for f in LOCAL_TRACKING_STAGING_DIR.glob("*.json"):
            try:
                f.unlink()
            except OSError:
                pass

        # Copy this batch's JSONs from Google Drive to local staging
        local_paths = {}
        for sc_id, remote_fp in batch:
            dst = LOCAL_TRACKING_STAGING_DIR / os.path.basename(remote_fp)
            try:
                shutil.copy2(remote_fp, dst)
                local_paths[sc_id] = dst
            except Exception as e:
                print(f"⚠️  Could not copy {remote_fp} to local staging: {e}")
                skipped += 1

        # Process each match from local file
        for sc_id, remote_fp in batch:
            local_fp = local_paths.get(sc_id)
            if local_fp is None:
                continue

            print(f"Processing match {sc_id}: {os.path.basename(remote_fp)} (local copy)")
            merged_df, out_sc_id = process_tracking_json_and_merge(
                str(local_fp),
                events_df=events_df,
                match_mapping=match_mapping,
                players_df=players_df,
                subsample_seconds=subsample_seconds,
                priority_events=priority_events,
                preserve_all_frames=preserve_all_frames,
            )
            if merged_df is None:
                skipped += 1
                continue

            out_fp = OUTPUT_MATCH_DIR / f"match_{out_sc_id}.parquet"
            merged_df.to_parquet(out_fp, index=False, compression="snappy")
            processed += 1
            del merged_df

    print(f"\nDone. Processed={processed}, skipped={skipped}. Outputs in {OUTPUT_MATCH_DIR}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge SkillCorner tracking JSON with StatsBomb events, output one parquet per match.")
    parser.add_argument("--subsample", type=float, default=0.2, help="Tracking subsample interval in seconds (default: 0.2).")
    parser.add_argument("--priority-events", nargs="+", default=["Pass", "Carry", "Shot"], help="Priority event types for conflict resolution.")
    parser.add_argument("--force-reprocess", action="store_true", help="Recreate outputs even if match parquet already exists.")
    parser.add_argument("--mapping-overwrite", action="store_true", help="Rebuild mapping CSV even if it already exists.")
    parser.add_argument("--mapping-only", action="store_true", help="Only build mapping CSV, do not merge tracking with events.")
    parser.add_argument("--preserve-all-frames", action="store_true", help="Do not subsample tracking; keep all frames (slow, large).")
    parser.add_argument("--match-id", type=int, action="append", help="Only process a specific SkillCorner match id (can be repeated).")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of matches to stage locally at once (default: 20).")

    args = parser.parse_args()

    main(
        subsample_seconds=args.subsample,
        priority_events=args.priority_events,
        force_reprocess=args.force_reprocess,
        mapping_overwrite=args.mapping_overwrite,
        mapping_only=args.mapping_only,
        preserve_all_frames=args.preserve_all_frames,
        match_ids=args.match_id,
        batch_size=args.batch_size,
    )