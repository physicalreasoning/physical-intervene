"""Checks on the repo's docs, run in CI (.github/workflows/docs.yml).

links:  every relative Markdown link in README.md and docs/ points to a file that exists.
dashes: no em dashes in docs, code or workflows (the project's writing style).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)\s]+)\)")
EM_DASH = "\u2014"


def markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def text_files() -> list[Path]:
    files = markdown_files()
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("scripts", "*.py"), (".github", "*.yml")):
        files += sorted((ROOT / folder).rglob(pattern))
    return files


def check_links() -> list[str]:
    errors = []
    for md in markdown_files():
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for target in LINK.findall(line):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                if path and not (md.parent / path).exists():
                    errors.append(f"{md.relative_to(ROOT)}:{lineno}: broken link {target}")
    return errors


def check_dashes() -> list[str]:
    errors = []
    for f in text_files():
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if EM_DASH in line:
                errors.append(f"{f.relative_to(ROOT)}:{lineno}: em dash")
    return errors


def main() -> int:
    checks = {"links": check_links, "dashes": check_dashes}
    names = sys.argv[1:] or list(checks)
    errors = [e for name in names for e in checks[name]()]
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
