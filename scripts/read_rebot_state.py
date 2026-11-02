"""One read-only encoder snapshot as JSON. No SDK connect or motor commands."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from cascade.control.rebot_observer import read_snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-config", required=True, type=Path)
    parser.add_argument("--limits-json", required=True)
    parser.add_argument("--settings", action="store_true",
                        help="Also read motor mode/limits/gains; never write parameters")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Also read command/current/fault registers; zeros do not establish support")
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            result = read_snapshot(args.sdk_config, json.loads(args.limits_json),
                                   include_settings=args.settings, include_diagnostics=args.diagnostics)
        print(json.dumps(result, allow_nan=False))
        return 2 if result.get("settings_complete") is False else 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:240]}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
