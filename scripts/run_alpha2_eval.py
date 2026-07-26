"""
Alpha v2 walk-forward evaluation: survivor-complete season-points residual,
add-one feature selection, per-player shrinkage. Writes
output/experiments/alpha_v2/{config.json, metrics.json, predictions.parquet}.

Usage: .venv/bin/python scripts/run_alpha2_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import TRAINING_SEASONS                                  # noqa: E402
from data.adp import load_adp                                        # noqa: E402
from data.loader import load_weekly                                  # noqa: E402
from features.assembler import assemble_feature_matrix, build_yoy_pairs  # noqa: E402
from models.two_stage import ALL_RATE_TARGET_COLS                    # noqa: E402
from models import alpha2                                            # noqa: E402

OUT = ROOT / "output" / "experiments" / "alpha_v2"


def main() -> None:
    fm = assemble_feature_matrix(TRAINING_SEASONS)
    weekly = load_weekly(list(TRAINING_SEASONS))
    adp = load_adp(list(range(2013, max(TRAINING_SEASONS) + 2)))
    pairs_old = build_yoy_pairs(fm, extra_target_cols=ALL_RATE_TARGET_COLS)

    frame = alpha2.build_survivor_frame(fm, weekly, adp)
    priced = frame["adp"].notna()
    print(f"survivor frame: {len(frame)} rows, {int(priced.sum())} priced; "
          f"status: {frame.loc[priced, 'outcome_status'].value_counts().to_dict()}")

    features, selection, fund_df = alpha2.select_features(frame, pairs_old)
    wf, _ = alpha2.run_alpha_v2_walkforward(frame, pairs_old, features=features,
                                            fund_df=fund_df)
    metrics = alpha2.evaluate_walkforward(wf)
    metrics["feature_selection"] = selection

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(
        {"target": alpha2.TARGET, "features": features,
         "base": alpha2.BASE_FEATURES, "candidates": alpha2.CANDIDATE_FEATURES},
        indent=2))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    keep = [c for c in ["player_id", "player_name", "season", "position", "adp",
                        "adp_stdev", "market_expected", "fund_season", "residual",
                        "pred_residual", "lam_i", "fair", alpha2.TARGET,
                        "outcome_status"] if c in wf.columns]
    wf[keep].to_parquet(OUT / "predictions.parquet")

    print(json.dumps({k: metrics[k] for k in ["pooled", "ls_mean_spread"]}, indent=2))
    print("per-season:", [(r["season"], r["residual_ic"]) for r in metrics["per_season"]])
    print("buckets:", [(r["bucket"], r["residual_ic"]) for r in metrics["by_adp_bucket"]])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
