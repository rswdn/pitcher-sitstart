# Pitcher Sit/Start Analyzer

Fantasy streaming helper that pulls probable starters for a given date, enriches them with recent performance, matchup context, and park factors, and produces a CSV with a start/stream/sit recommendation.

## Features
- Fetch probable starters via `pybaseball`.
- Enrich with recent rolling stats (last 5 starts), season-to-date rates, opponent platoon splits, and park factors.
- Compute a simple stream score and tier (Start/Stream/Sit).
- Export per-date CSVs.

## Setup
The base image here is missing `pip`/`ensurepip`. If `python3 -m venv` fails with an `ensurepip` error, install the venv tooling first (e.g. `sudo apt install python3-venv` or `sudo apt install python3.12-venv`). The user requested `virtualenv`; after `pip` works, install it with `python3 -m pip install --user virtualenv` and create the env with `python3 -m virtualenv venv`.

Once venv tooling is available:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
source venv/bin/activate
python src/score_today.py --date 2024-06-15 --out data/sit_start_2024-06-15.csv
```

## Notes
- The initial model is heuristic; swap in a trained model later via `model.py`.
- Caching for API pulls can be added by persisting raw data in `data/`.
