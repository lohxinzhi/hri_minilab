#!/usr/bin/env python3
"""Run the deterministic target parser on one quoted command."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minilab.dialogue.parser import parse_target_command  # noqa: E402


def main():
    text = " ".join(sys.argv[1:])
    result = parse_target_command(text)
    print(f"status={result.status}")
    print(f"action={result.action or 'null'}")
    print(f"object_class={result.object_class or 'null'}")
    print(f"color={result.color or 'null'}")
    print(f"reason={result.reason}")


if __name__ == "__main__":
    main()
