"""Data fetching and feature assembly for the sit/start analyzer."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

# Ensure a browser-like UA is present before any pybaseball requests.
os.environ.setdefault(
    "PYBASEBALL_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
)

from pybaseball import statcast_pitcher

try:
    # Set a browser-like user agent to reduce 403s from pybaseball endpoints.
    from pybaseball.datahelpers import general as _pb_general

    _pb_general.set_user_agent(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
except Exception:
    pass

from config import CACHE_DIR, PARK_FACTORS_PATH


def _parse_date(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()


def _resolve_probable_pitchers_statsapi(target_date: dt.date) -> pd.DataFrame:
    """Resolve probable starters using MLB Stats API (most reliable)."""
    date_str = target_date.strftime("%Y-%m-%d")
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,teams",
    }

    team_map = _team_id_to_abbr_map()

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    dates = payload.get("dates", [])
    if not dates:
        return pd.DataFrame()

    rows = []
    for game in dates[0].get("games", []):
        game_date = _parse_date(game.get("officialDate", date_str))

        home = game["teams"]["home"]
        away = game["teams"]["away"]

        home_team = home["team"]
        away_team = away["team"]

        home_abbr = team_map.get(home_team.get("id"))
        away_abbr = team_map.get(away_team.get("id"))

        home_prob = home.get("probablePitcher")
        away_prob = away.get("probablePitcher")

        if home_prob:
            rows.append(
                {
                    "pitcher": home_prob.get("fullName"),
                    "pitcher_id": home_prob.get("id"),
                    "team": home_abbr,
                    "opp": away_abbr,
                    "hand": home_prob.get("pitchHand", {}).get("code")
                    or _pitcher_hand(home_prob.get("id")),
                    "home": True,
                    "game_date": game_date,
                }
            )

        if away_prob:
            rows.append(
                {
                    "pitcher": away_prob.get("fullName"),
                    "pitcher_id": away_prob.get("id"),
                    "team": away_abbr,
                    "opp": home_abbr,
                    "hand": away_prob.get("pitchHand", {}).get("code")
                    or _pitcher_hand(away_prob.get("id")),
                    "home": False,
                    "game_date": game_date,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and df["team"].isna().all():
        raise RuntimeError(
            "Probable starters found but team abbreviations are missing. "
            "Check MLB Stats API team mapping response."
        )
    return df


def _resolve_probable_pitchers(target_date: dt.date) -> pd.DataFrame:
    # Primary: MLB Stats API
    try:
        df = _resolve_probable_pitchers_statsapi(target_date)
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: pybaseball
    try:
        df = _resolve_probable_pitchers_pybaseball(target_date)
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame()


def _resolve_probable_pitchers_pybaseball(target_date: dt.date) -> pd.DataFrame:
    """Resolve probable starters using pybaseball (best-effort fallback)."""
    probable_fn = None
    for mod_path in (
        "pybaseball.probables",
        "pybaseball.schedule_and_lineup",
        "pybaseball",
    ):
        try:
            mod = __import__(mod_path, fromlist=["probable_pitchers"])
            probable_fn = getattr(mod, "probable_pitchers", None)
            if probable_fn:
                break
        except ImportError:
            continue

    if not probable_fn:
        return pd.DataFrame()

    raw = probable_fn(target_date)
    pitchers = []

    for _, row in raw.iterrows():
        game_date = _parse_date(str(row.get("game_date", target_date)))

        pitchers.append(
            {
                "pitcher": row.get("probable_pitcher_home"),
                "pitcher_id": row.get("probable_pitcher_home_id"),
                "team": row.get("home_team"),
                "opp": row.get("away_team"),
                "hand": row.get("probable_pitcher_home_hand"),
                "home": True,
                "game_date": game_date,
            }
        )

        pitchers.append(
            {
                "pitcher": row.get("probable_pitcher_away"),
                "pitcher_id": row.get("probable_pitcher_away_id"),
                "team": row.get("away_team"),
                "opp": row.get("home_team"),
                "hand": row.get("probable_pitcher_away_hand"),
                "home": False,
                "game_date": game_date,
            }
        )

    return (
        pd.DataFrame(pitchers)
        .dropna(subset=["pitcher"])
        .reset_index(drop=True)
    )


def fetch_probable_starters(target_date: dt.date) -> pd.DataFrame:
    """Return probable starters for a given date, normalized to one row per pitcher."""
    df = _resolve_probable_pitchers(target_date)
    return df.reset_index(drop=True)


def _team_id_to_abbr_map() -> Dict[int, str]:
    cache_path = _cache_path("mlb_teams_map.csv")
    cached = _read_cache(cache_path)
    if cached is not None and {"id", "abbreviation"}.issubset(cached.columns):
        return {
            int(row["id"]): row["abbreviation"]
            for _, row in cached.iterrows()
            if pd.notna(row.get("id")) and pd.notna(row.get("abbreviation"))
        }

    url = "https://statsapi.mlb.com/api/v1/teams"
    params = {"sportId": 1}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    teams = payload.get("teams", [])

    rows = []
    for team in teams:
        team_id = team.get("id")
        abbr = team.get("abbreviation")
        if team_id is None or not abbr:
            continue
        rows.append({"id": team_id, "abbreviation": abbr})

    team_df = pd.DataFrame(rows)
    _write_cache(cache_path, team_df)
    return {int(row["id"]): row["abbreviation"] for _, row in team_df.iterrows()}


def _pitcher_hand(pitcher_id: int) -> Optional[str]:
    if pitcher_id is None:
        return None
    cache_path = _cache_path("mlb_pitcher_handedness.csv")
    cached = _read_cache(cache_path)
    if cached is not None and {"pitcher_id", "hand"}.issubset(cached.columns):
        match = cached[cached["pitcher_id"] == pitcher_id]
        if not match.empty:
            hand = match.iloc[0].get("hand")
            return hand if pd.notna(hand) else None

    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        people = payload.get("people", [])
        if not people:
            return None
        hand = people[0].get("pitchHand", {}).get("code")
        if not hand:
            return None
    except Exception:
        return None

    new_row = pd.DataFrame([{"pitcher_id": pitcher_id, "hand": hand}])
    if cached is not None and {"pitcher_id", "hand"}.issubset(cached.columns):
        updated = pd.concat([cached, new_row], ignore_index=True)
        updated = updated.drop_duplicates(subset=["pitcher_id"], keep="last")
    else:
        updated = new_row
    _write_cache(cache_path, updated)
    return hand


def _statcast_cache_path(pitcher_id: int, start_date: dt.date, end_date: dt.date) -> Path:
    filename = f"statcast_pitcher_{pitcher_id}_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}.csv"
    return _cache_path(filename)


def fetch_statcast_window(
    pitcher_id: int, start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:
    cache_path = _statcast_cache_path(pitcher_id, start_date, end_date)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    data = statcast_pitcher(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        pitcher_id,
    )
    _write_cache(cache_path, data)
    return data


def load_park_factors() -> Dict[str, Dict[str, float]]:
    with open(PARK_FACTORS_PATH, "r", encoding="ascii") as f:
        return json.load(f)


def attach_pitcher_features(
    starters: pd.DataFrame, target_date: dt.date, local_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Join recent Statcast pitcher features and park factors."""
    recent_end = target_date - dt.timedelta(days=1)
    recent_start = recent_end - dt.timedelta(days=29)
    park_factors = load_park_factors()

    features = []
    for _, row in starters.iterrows():
        pitcher_id = row.get("pitcher_id")
        statcast = pd.DataFrame()
        if pd.notna(pitcher_id):
            try:
                statcast = fetch_statcast_window(int(pitcher_id), recent_start, recent_end)
            except Exception:
                statcast = pd.DataFrame()
        park = park_factors.get(row["team"], {"run": 1.0, "hr": 1.0})

        pa_recent = None
        k_pct_recent = None
        bb_pct_recent = None
        k_bb_pct_recent = None
        avg_velo_recent = None
        avg_ev_recent = None

        if not statcast.empty:
            pa_rows = statcast[statcast["events"].notna()]
            pa_recent = int(len(pa_rows))
            if pa_recent > 0:
                k_events = pa_rows["events"].str.contains("strikeout", na=False)
                bb_events = pa_rows["events"].isin(["walk", "intent_walk"])
                k_pct_recent = float(k_events.sum()) / pa_recent
                bb_pct_recent = float(bb_events.sum()) / pa_recent
                k_bb_pct_recent = k_pct_recent - bb_pct_recent

            if "release_speed" in statcast.columns:
                avg_velo_recent = statcast["release_speed"].mean()
            if "launch_speed" in statcast.columns:
                avg_ev_recent = statcast["launch_speed"].mean()

        features.append(
            {
                **row.to_dict(),
                "pa_recent": pa_recent,
                "k_pct_recent": k_pct_recent,
                "bb_pct_recent": bb_pct_recent,
                "k_bb_pct_recent": k_bb_pct_recent,
                "avg_velo_recent": avg_velo_recent,
                "avg_ev_recent": avg_ev_recent,
                "park_run_factor": park.get("run", 1.0),
                "park_hr_factor": park.get("hr", 1.0),
            }
        )

    df = pd.DataFrame(features)
    core_cols = [
        "k_bb_pct_recent",
        "k_pct_recent",
        "bb_pct_recent",
        "avg_velo_recent",
        "avg_ev_recent",
        "park_run_factor",
    ]
    for col in core_cols:
        if col not in df.columns:
            df[col] = pd.NA
    present_list = []
    present_counts = []
    for _, row in df.iterrows():
        present_cols = [col for col in core_cols if pd.notna(row.get(col))]
        present_list.append(",".join(present_cols))
        present_counts.append(len(present_cols))
    df["features_present"] = pd.Series(present_counts, index=df.index, dtype="int64")
    df["features_missing"] = len(core_cols) - df["features_present"]
    df["features_present_list"] = present_list
    return df


def build_feature_table(
    date_str: str, skip_fangraphs: bool = False, local_fangraphs_dir: Optional[Path] = None
) -> pd.DataFrame:
    target_date = _parse_date(date_str)
    starters = fetch_probable_starters(target_date)
    enriched = attach_pitcher_features(starters, target_date, local_dir=local_fangraphs_dir)
    return enriched


def _cache_path(filename: str) -> Path:
    return CACHE_DIR / filename


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        # Cache is best-effort; ignore failures.
        return
