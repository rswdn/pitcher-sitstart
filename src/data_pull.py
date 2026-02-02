"""Data fetching and feature assembly for the sit/start analyzer."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
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

from pybaseball import batting_stats_range, pitching_stats, pitching_stats_range

try:
    # Set a browser-like user agent to reduce 403s from FanGraphs endpoints.
    from pybaseball.datahelpers import general as _pb_general

    _pb_general.set_user_agent(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
except Exception:
    pass

from config import CACHE_DIR, FG_LOCAL_DIR, PARK_FACTORS_PATH, RECENT_STARTS_WINDOW


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

        home_abbr = (
            home_team.get("abbreviation")
            or home_team.get("teamCode")
            or home_team.get("triCode")
        )
        away_abbr = (
            away_team.get("abbreviation")
            or away_team.get("teamCode")
            or away_team.get("triCode")
        )

        home_prob = home.get("probablePitcher")
        away_prob = away.get("probablePitcher")

        if home_prob:
            rows.append(
                {
                    "pitcher": home_prob.get("fullName"),
                    "pitcher_id": home_prob.get("id"),
                    "team": home_abbr,
                    "opp": away_abbr,
                    "hand": home_prob.get("pitchHand", {}).get("code"),
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
                    "hand": away_prob.get("pitchHand", {}).get("code"),
                    "home": False,
                    "game_date": game_date,
                }
            )

    return pd.DataFrame(rows)

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


    date_str = target_date.strftime("%Y-%m-%d")
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str, "hydrate": "probablePitcher"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    dates = payload.get("dates", [])
    games = dates[0].get("games", []) if dates else []
    rows = []
    for game in games:
        game_date = _parse_date(game.get("officialDate", date_str))
        home_team = game["teams"]["home"]["team"]
        away_team = game["teams"]["away"]["team"]
        home_abbr = home_team.get("abbreviation") or home_team.get("teamCode") or home_team.get("triCode")
        away_abbr = away_team.get("abbreviation") or away_team.get("teamCode") or away_team.get("triCode")
        home_prob = game["teams"]["home"].get("probablePitcher") or {}
        away_prob = game["teams"]["away"].get("probablePitcher") or {}
        if home_prob.get("fullName"):
            rows.append(
                {
                    "pitcher": home_prob.get("fullName"),
                    "pitcher_id": home_prob.get("id"),
                    "team": home_abbr,
                    "opp": away_abbr,
                    "hand": None,
                    "home": True,
                    "game_date": game_date,
                }
            )
        if away_prob.get("fullName"):
            rows.append(
                {
                    "pitcher": away_prob.get("fullName"),
                    "pitcher_id": away_prob.get("id"),
                    "team": away_abbr,
                    "opp": home_abbr,
                    "hand": None,
                    "home": False,
                    "game_date": game_date,
                }
            )
    return pd.DataFrame(rows)


def fetch_probable_starters(target_date: dt.date) -> pd.DataFrame:
    """Return probable starters for a given date, normalized to one row per pitcher."""
    df = _resolve_probable_pitchers(target_date)
    return df.reset_index(drop=True)


def fetch_season_stats(year: int, local_dir: Optional[Path] = None) -> pd.DataFrame:
    """Season-to-date stats from FanGraphs."""
    local = _read_local_csv(local_dir, "season_pitching.csv")
    if local is not None:
        return local
    cache_path = _cache_path(f"season_{year}.csv")
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    stats = _retry_pybaseball(pitching_stats, year, qual=0)
    stats = stats.rename(columns={"IDfg": "player_id"})
    _write_cache(cache_path, stats)
    return stats


def fetch_recent_stats(
    start_date: dt.date, end_date: dt.date, local_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Rolling window stats using FanGraphs range endpoint."""
    local = _read_local_csv(local_dir, "recent_pitching.csv")
    if local is not None:
        return local
    cache_path = _cache_path(f"recent_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}.csv")
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    stats = _retry_pybaseball(
        pitching_stats_range,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        qual=0,
    )
    stats = stats.rename(columns={"IDfg": "player_id"})
    _write_cache(cache_path, stats)
    return stats


def fetch_opponent_offense(
    start_date: dt.date, end_date: dt.date, local_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Opponent offense stats over a recent range (defaults to last 30 days)."""
    local = _read_local_csv(local_dir, "offense.csv")
    if local is not None:
        return local
    cache_path = _cache_path(f"offense_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}.csv")
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    bats = _retry_pybaseball(
        batting_stats_range, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
    bats = bats.rename(columns={"Team": "team"})
    team_cols = [
        "team",
        "OPS",
        "wRC+",
        "K%",
        "BB%",
    ]
    bats = bats[team_cols]
    _write_cache(cache_path, bats)
    return bats


def _retry_pybaseball(func, *args, **kwargs) -> pd.DataFrame:
    """Retry wrapper to handle transient FanGraphs 403s."""
    last_exc = None
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - best-effort retry for upstream failures.
            last_exc = exc
            time.sleep(2 ** attempt)
    raise last_exc


def load_park_factors() -> Dict[str, Dict[str, float]]:
    with open(PARK_FACTORS_PATH, "r", encoding="ascii") as f:
        return json.load(f)


def attach_pitcher_features(
    starters: pd.DataFrame, target_date: dt.date, local_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Join season stats, recent stats, opponent context, and park factors."""
    season_stats = fetch_season_stats(target_date.year, local_dir=local_dir)
    recent_start = target_date - dt.timedelta(days=30)
    recent_stats = fetch_recent_stats(recent_start, target_date, local_dir=local_dir)
    opp_offense = fetch_opponent_offense(recent_start, target_date, local_dir=local_dir)
    park_factors = load_park_factors()

    def pick_row(df: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
        matches = df[df["Name"] == player_name]
        if matches.empty:
            return None
        return matches.iloc[0]

    features = []
    for _, row in starters.iterrows():
        season_row = pick_row(season_stats, row["pitcher"])
        recent_row = pick_row(recent_stats, row["pitcher"])
        opp_row = opp_offense[opp_offense["team"] == row["opp"]]
        opp = opp_row.iloc[0] if not opp_row.empty else None
        park = park_factors.get(row["team"], {"run": 1.0, "hr": 1.0})

        def val(src: Optional[pd.Series], key: str) -> Optional[float]:
            if src is None:
                return None
            return src.get(key)

        features.append(
            {
                **row.to_dict(),
                "siera": val(season_row, "SIERA"),
                "xfip": val(season_row, "xFIP"),
                "k_bb_pct": val(season_row, "K-BB%"),
                "csw_pct": val(season_row, "CSW%"),
                "whiff_pct": val(recent_row, "SwStr%"),
                "gb_pct": val(season_row, "GB%"),
                "barrel_pa_pct": val(season_row, "Barrel%"),
                "opp_wrc_plus_vs_hand": opp["wRC+"] if opp is not None else None,
                "opp_k_pct_vs_hand": opp["K%"] if opp is not None else None,
                "park_run_factor": park.get("run", 1.0),
                "park_hr_factor": park.get("hr", 1.0),
            }
        )

    df = pd.DataFrame(features)
    return df


def attach_minimal_features(starters: pd.DataFrame) -> pd.DataFrame:
    """Attach only park factors when FanGraphs is unavailable."""
    park_factors = load_park_factors()
    features = []
    for _, row in starters.iterrows():
        park = park_factors.get(row["team"], {"run": 1.0, "hr": 1.0})
        features.append(
            {
                **row.to_dict(),
                "park_run_factor": park.get("run", 1.0),
                "park_hr_factor": park.get("hr", 1.0),
            }
        )
    return pd.DataFrame(features)


def build_feature_table(
    date_str: str, skip_fangraphs: bool = False, local_fangraphs_dir: Optional[Path] = None
) -> pd.DataFrame:
    target_date = _parse_date(date_str)
    starters = fetch_probable_starters(target_date)
    if skip_fangraphs:
        enriched = attach_minimal_features(starters)
    else:
        enriched = attach_pitcher_features(
            starters, target_date, local_dir=local_fangraphs_dir
        )
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


def _read_local_csv(local_dir: Optional[Path], filename: str) -> Optional[pd.DataFrame]:
    if local_dir is None:
        return None
    path = local_dir / filename
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # Normalize common FanGraphs fields if present.
    if "IDfg" in df.columns:
        df = df.rename(columns={"IDfg": "player_id"})
    if "Team" in df.columns:
        df = df.rename(columns={"Team": "team"})
    return df
