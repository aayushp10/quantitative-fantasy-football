"""
Audit the training-target universe: how many drafted players does the old
year-over-year pair construction silently drop, and why?

For each season S with an ADP archive, take the drafted universe (ADP
top-150 of season S+1 drafts, joined to season-S feature rows) and count:
  - predictors: drafted players with a season-S feature row
  - old_pairs:  those surviving build_yoy_pairs (threshold-filtered shift)
  - new_pairs:  those in survivor-complete pairs (always all predictors)
  - zero_games / low_usage / played: their true N+1 outcomes
The old-method loss rate is the survivorship bias the alpha model was
trained under before v2.

Usage: .venv/bin/python scripts/audit_target_universe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import TRAINING_SEASONS                       # noqa: E402
from features.assembler import assemble_feature_matrix, build_yoy_pairs  # noqa: E402
from data.loader import load_weekly                        # noqa: E402
from data.adp import load_adp                              # noqa: E402
from data.outcomes import build_outcomes                   # noqa: E402
from data.training_pairs import build_training_pairs       # noqa: E402
from models.alpha import attach_market                     # noqa: E402

ADP_TOP_N = 150


def main() -> None:
    fm = assemble_feature_matrix(TRAINING_SEASONS)
    weekly = load_weekly(list(TRAINING_SEASONS))
    outcomes = build_outcomes(weekly)

    old_pairs = build_yoy_pairs(fm)
    new_pairs = build_training_pairs(fm, outcomes)
    adp = load_adp(list(range(2013, max(TRAINING_SEASONS) + 2)))

    new_adp = attach_market(new_pairs, adp)
    old_keys = set(zip(old_pairs["player_id"], old_pairs["season"]))

    rows = []
    for season, grp in new_adp.groupby("season"):
        drafted = grp[grp["adp"].notna() & (grp["adp"] <= ADP_TOP_N)]
        drafted = drafted.drop_duplicates(subset=["player_id"])
        if drafted.empty:
            continue
        in_old = drafted.apply(
            lambda r: (r["player_id"], r["season"]) in old_keys, axis=1)
        status = drafted["outcome_status"].value_counts()
        rows.append({
            "season": int(season),
            "drafted_with_features": len(drafted),
            "in_old_pairs": int(in_old.sum()),
            "dropped_by_old": int((~in_old).sum()),
            "drop_rate": round(float((~in_old).mean()), 3),
            "no_games": int(status.get("no_games", 0)),
            "low_usage": int(status.get("low_usage", 0)),
            "played": int(status.get("played", 0)),
            "mean_pts_dropped": round(
                float(drafted.loc[~in_old, "next_season_points"].mean()), 1)
            if (~in_old).any() else None,
            "mean_pts_kept": round(
                float(drafted.loc[in_old, "next_season_points"].mean()), 1),
        })

    df = pd.DataFrame(rows)
    print("\nDrafted universe (ADP top-%d) vs old training pairs\n" % ADP_TOP_N)
    print(df.to_string(index=False))
    tot = df[["drafted_with_features", "in_old_pairs", "dropped_by_old"]].sum()
    print(f"\nTOTAL: {tot['dropped_by_old']}/{tot['drafted_with_features']} drafted players "
          f"({tot['dropped_by_old']/tot['drafted_with_features']:.1%}) were invisible to "
          "old training — their outcomes never entered the target.")


if __name__ == "__main__":
    main()
