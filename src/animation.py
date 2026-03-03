"""
Tracking data animation utilities.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .pitch import create_pitch
from .player_mapping import get_player_team_mapping, split_players_by_team


def create_tracking_animation(
    file_path,
    players_df,
    start_frame=0,
    end_frame=100,
    save_path=None,
    fps=20,
    interval=20,
):
    """
    Create an animation from tracking data.

    Parameters
    ----------
    file_path : str
        Path to tracking JSON file
    players_df : pandas.DataFrame
        DataFrame with player-team mappings
    start_frame, end_frame : int
        Frame range to animate
    save_path : str or None
        If provided, save animation to this path
    fps : int
        Frames per second for saved video
    interval : int
        Milliseconds between frames in animation

    Returns
    -------
    tuple
        (fig, anim) - matplotlib Figure and FuncAnimation objects
    """
    # Load tracking data
    with open(file_path, "r") as f:
        frames = json.load(f)

    frames_slice = frames[start_frame : end_frame + 1]

    # Get player-team mapping
    player_to_team, home_team_id, away_team_id, match_id = get_player_team_mapping(
        file_path, players_df
    )

    # Create figure and pitch
    fig, ax = plt.subplots(figsize=(14, 10))
    create_pitch(ax, title=f"Match {match_id} – Frames {start_frame}–{end_frame}")

    # Initialize scatter plots
    away_scatter = ax.scatter(
        [], [],
        c="red",
        s=80,
        label="Away Team",
        edgecolors="white",
        linewidths=1,
        zorder=5,
        alpha=0.9,
    )
    home_scatter = ax.scatter(
        [], [],
        c="blue",
        s=80,
        label="Home Team",
        edgecolors="white",
        linewidths=1,
        zorder=5,
        alpha=0.9,
    )
    ball_scatter = ax.scatter(
        [], [],
        c="black",
        s=100,
        label="Ball",
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        alpha=0.95,
    )

    timestamp_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=13,
        weight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="black",
            alpha=0.8,
            edgecolor="white",
            linewidth=1.5,
        ),
    )

    ax.legend(
        loc="upper right",
        facecolor="white",
        framealpha=0.95,
        fontsize=11,
        edgecolor="black",
        frameon=True,
    )

    # Animation update function
    def update(i):
        frame = frames_slice[i]
        player_data = frame.get("player_data", [])
        (away_x, away_y), (home_x, home_y) = split_players_by_team(
            player_data, player_to_team, home_team_id, away_team_id
        )

        # Update scatter plots
        if away_x:
            away_scatter.set_offsets(np.column_stack([away_x, away_y]))
        else:
            away_scatter.set_offsets(np.empty((0, 2)))

        if home_x:
            home_scatter.set_offsets(np.column_stack([home_x, home_y]))
        else:
            home_scatter.set_offsets(np.empty((0, 2)))

        # Update ball
        ball = frame.get("ball_data", {})
        if (
            ball.get("is_detected")
            and ball.get("x") is not None
            and ball.get("y") is not None
        ):
            ball_scatter.set_offsets(np.array([[ball["x"], ball["y"]]], dtype=float))
            ball_scatter.set_visible(True)
        else:
            ball_scatter.set_offsets(np.empty((0, 2)))
            ball_scatter.set_visible(False)

        # Update timestamp
        ts = frame.get("timestamp", "N/A")
        period = frame.get("period", "N/A")
        frame_num = start_frame + i
        timestamp_text.set_text(
            f"Frame: {frame_num} | Period: {period} | Time: {ts}\n"
            f"Away: {len(away_x)} players | Home: {len(home_x)} players"
        )

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames_slice),
        interval=interval,
        repeat=True,
        blit=False,
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="ffmpeg", fps=fps, bitrate=1800)
        print("Animation saved!")

    return fig, anim


def create_tracking_animation_from_df(
    df,
    match_id=None,
    start_frame=None,
    end_frame=None,
    save_path=None,
    fps=20,
    interval=20,
):
    """
    Create an animation from a long-format tracking DataFrame
    (e.g. merged match_<id>.parquet).

    Expected columns:
      - frame_number, period, timestamp (seconds), x, y, team
      - ball_x, ball_y, ball_z (optional), ball_is_detected (optional)
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Restrict to frame range if provided
    if start_frame is not None:
        df = df[df["frame_number"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame_number"] <= end_frame]

    if df.empty:
        raise ValueError("No data in specified frame range")

    # Sort by frame/time
    df = df.sort_values(["frame_number", "timestamp"])

    # Infer match_id and frame range if not given
    if match_id is None and "skillcorner_match_id" in df.columns:
        match_id = int(df["skillcorner_match_id"].iloc[0])

    frame_numbers = sorted(df["frame_number"].unique().tolist())
    first_frame = frame_numbers[0]
    last_frame = frame_numbers[-1]

    # Infer team IDs (SkillCorner team ids)
    team_ids = sorted(df["team"].dropna().unique().tolist())
    if len(team_ids) != 2:
        raise ValueError(f"Expected exactly 2 team ids, found {team_ids}")
    home_team_id, away_team_id = team_ids  # arbitrary but consistent

    # Build per-frame views for faster access in update()
    frames_dict = {fn: df[df["frame_number"] == fn] for fn in frame_numbers}

    # Figure and pitch
    fig, ax = plt.subplots(figsize=(14, 10))
    title = f"Match {match_id} – Frames {first_frame}–{last_frame}"
    create_pitch(ax, title=title)

    # Scatter plots
    away_scatter = ax.scatter(
        [],
        [],
        c="red",
        s=80,
        label="Away Team",
        edgecolors="white",
        linewidths=1,
        zorder=5,
        alpha=0.9,
    )
    home_scatter = ax.scatter(
        [],
        [],
        c="blue",
        s=80,
        label="Home Team",
        edgecolors="white",
        linewidths=1,
        zorder=5,
        alpha=0.9,
    )
    ball_scatter = ax.scatter(
        [],
        [],
        c="black",
        s=100,
        label="Ball",
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        alpha=0.95,
    )

    timestamp_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=13,
        weight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="black",
            alpha=0.8,
            edgecolor="white",
            linewidth=1.5,
        ),
    )

    ax.legend(
        loc="upper right",
        facecolor="white",
        framealpha=0.95,
        fontsize=11,
        edgecolor="black",
        frameon=True,
    )

    def update(i):
        frame_num = frame_numbers[i]
        fdf = frames_dict[frame_num]

        # Split by team
        home = fdf[fdf["team"] == home_team_id]
        away = fdf[fdf["team"] == away_team_id]

        away_xy = away[["x", "y"]].dropna().to_numpy()
        home_xy = home[["x", "y"]].dropna().to_numpy()

        away_scatter.set_offsets(away_xy if len(away_xy) else np.empty((0, 2)))
        home_scatter.set_offsets(home_xy if len(home_xy) else np.empty((0, 2)))

        # Ball (if present)
        if {"ball_x", "ball_y"}.issubset(fdf.columns):
            ball_row = fdf.dropna(subset=["ball_x", "ball_y"]).head(1)
            if not ball_row.empty:
                ball_xy = ball_row[["ball_x", "ball_y"]].to_numpy()
                ball_scatter.set_offsets(ball_xy)
                ball_scatter.set_visible(True)
            else:
                ball_scatter.set_offsets(np.empty((0, 2)))
                ball_scatter.set_visible(False)
        else:
            ball_scatter.set_offsets(np.empty((0, 2)))
            ball_scatter.set_visible(False)

        period = fdf["period"].iloc[0] if "period" in fdf.columns else "N/A"
        ts = fdf["timestamp"].iloc[0] if "timestamp" in fdf.columns else None
        if ts is not None:
            minutes = int(ts // 60)
            seconds = ts - 60 * minutes
            ts_str = f"{minutes:02d}:{seconds:05.2f}"
        else:
            ts_str = "N/A"

        timestamp_text.set_text(
            f"Frame: {frame_num} | Period: {period} | Time: {ts_str}\n"
            f"Away ({away_team_id}): {len(away)} players | Home ({home_team_id}): {len(home)} players"
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_numbers),
        interval=interval,
        repeat=True,
        blit=False,
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="ffmpeg", fps=fps, bitrate=1800)
        print("Animation saved!")

    return fig, anim
