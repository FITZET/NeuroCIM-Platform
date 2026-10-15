from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True)
    args = parser.parse_args()
    if not args.client_id.strip():
        raise ValueError("GitHub OAuth client ID must not be empty")
    target = Path(__file__).resolve().parents[1] / "neurocim_opt" / "release_config.py"
    target.write_text(
        '"""Generated public release configuration."""\n\n'
        f'GITHUB_OAUTH_CLIENT_ID = {args.client_id.strip()!r}\n'
        'GITHUB_OWNER = "FITZET"\n'
        'GITHUB_DISTRIBUTION_REPOSITORY = "NeuroCIM-Distribution"\n'
        'ACCESS_PORTAL_URL = "https://fitzet.github.io/NeuroCIM-Access/"\n',
        encoding="utf-8",
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
