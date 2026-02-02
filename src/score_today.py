"""CLI entrypoint to score today's probable starters and export a CSV."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, FG_LOCAL_DIR
from data_pull import build_feature_table
from model import score_pitchers


def parse_args() -> argparse.Namespace:
    today = dt.date.today().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Score probable starters for sit/start decisions.")
    parser.add_argument("--date", default=today, help="Date to score (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--out", default=None, help="Output CSV path. Defaults to data/sit_start_<date>.csv")
    parser.add_argument(
        "--no-fg",
        action="store_true",
        help="Skip FanGraphs data pulls (use probable starters + park factors only).",
    )
    parser.add_argument(
        "--fg-local-dir",
        default=None,
        help=f"Use local FanGraphs CSVs from a directory (default template: {FG_LOCAL_DIR}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a debug sample of the feature table.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"sit_start_{args.date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    local_dir = Path(args.fg_local_dir) if args.fg_local_dir else None

    features = build_feature_table(
        args.date, skip_fangraphs=args.no_fg, local_fangraphs_dir=local_dir
    )

    if getattr(args, "debug", False):
        print("\nDEBUG: feature table sample")
        cols = [c for c in ["pitcher", "pitcher_id", "team", "opp", "hand", "home", "game_date"] if c in features.columns]
        print(features[cols].head(10))
        print("\nDEBUG: columns")
        print(list(features.columns))
        print("\nDEBUG: missing counts (top 15)")
        print(features.isna().sum().sort_values(ascending=False).head(15))

    scored = score_pitchers(features)
    scored.to_csv(out_path, index=False)
    print(f"Wrote {len(scored)} rows to {out_path}")


if __name__ == "__main__":
    main()
