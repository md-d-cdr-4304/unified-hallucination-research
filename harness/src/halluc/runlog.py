"""Result cards: one JSON per run with config, environment, metrics, timing.

Every experiment writes a result card so runs are comparable and auditable
(preregistration discipline: config recorded before metrics are computed).
"""

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


class ResultCard:
    def __init__(self, experiment: str, config: dict):
        self.card = {
            "experiment": experiment,
            "config": config,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "python": platform.python_version(),
            "metrics": {},
            "notes": [],
        }
        self._t0 = time.time()

    def note(self, msg: str):
        self.card["notes"].append(msg)

    def finish(self, metrics: dict) -> Path:
        self.card["metrics"] = metrics
        self.card["wall_seconds"] = round(time.time() - self._t0, 1)
        self.card["finished_utc"] = datetime.now(timezone.utc).isoformat()
        out_dir = RESULTS_DIR / self.card["experiment"]
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"card_{stamp}.json"
        path.write_text(json.dumps(self.card, indent=2))
        return path
